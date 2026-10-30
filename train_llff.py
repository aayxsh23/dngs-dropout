#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import time
import torch
import torchvision
from os import makedirs
from random import randint
from utils.graphics_utils import fov2focal
from utils.loss_utils import l1_loss, loss_depth_smoothness, patch_norm_mse_loss, patch_norm_mse_loss_global, ssim
# from utils.loss_utils import mssim as ssim
from gaussian_renderer import render, render_for_depth, render_for_opa  # , network_gui
import sys
from scene import RenderScene, Scene, GaussianModel
from utils.general_utils import safe_state, load_checkpoint
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    print('Launch TensorBoard')
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEPTH_RENDER_MODES = ("shared", "clean", "hard_only", "soft_only")
DIAG_EVERY = 100   # diag.csv row interval (PLAN W4)
DIAG_PATCH = 11    # fixed patch size (middle of DNGaussian's 5..17) so used and clean are comparable
DIAG_PROBE_RATE, DIAG_PROBES = 0.2, 4
DIAG_COLUMNS = ["iter", "loss_rgb", "loss_depth_used", "loss_depth_clean", "depth_var_masks", "n_gaussians",
                "drop_rate_now", "iter_ms", "vram_peak_mb", "rgb_var_masks"]


def depth_loss_diag(depth, depth_mono, it, tol):
    """The training depth loss of one pass, at a fixed patch size (0.1 local + 1 global + 0.1 smoothness after 3000)."""
    l = 0.1 * patch_norm_mse_loss(depth[None], depth_mono[None], DIAG_PATCH, tol) \
        + patch_norm_mse_loss_global(depth[None], depth_mono[None], DIAG_PATCH, tol)
    if it > 3000:
        l = l + 0.1 * loss_depth_smoothness(depth[None], depth_mono[None])
    return l


def depth_diag(cam, gaussians, pipe, background, it, opt, hard_comp, soft_comp):
    """Hard + soft depth loss on the renders as masked by hard_comp / soft_comp (None = clean)."""
    mono = 255.0 - cam.depth_mono
    v = depth_loss_diag(render_for_depth(cam, gaussians, pipe, background, drop_comp=hard_comp)["depth"], mono, it, opt.error_tolerance)
    if it > opt.soft_depth_start:
        v = v + depth_loss_diag(render_for_opa(cam, gaussians, pipe, background, drop_comp=soft_comp)["depth"], mono, it, opt.error_tolerance)
    return v.item()


def probe_masks(n, gen):
    """DIAG_PROBES compensated masks at DIAG_PROBE_RATE from the diagnostics generator (never the training RNG)."""
    keep = 1.0 - DIAG_PROBE_RATE
    return [(torch.rand(n, generator=gen, device="cuda") < keep).float() / keep for _ in range(DIAG_PROBES)]


def mask_variances(cam, gaussians, pipe, background, gen):
    """depth_var_masks (soft depth) and rgb_var_masks (photometric image): per-pixel variance across probe masks, mean over pixels."""
    masks = probe_masks(gaussians.get_xyz.shape[0], gen)
    depths = torch.stack([render_for_opa(cam, gaussians, pipe, background, drop_comp=m)["depth"] for m in masks])
    images = torch.stack([render(cam, gaussians, pipe, background, drop_comp=m)["render"] for m in masks])
    return depths.var(dim=0, unbiased=False).mean().item(), images.var(dim=0, unbiased=False).mean().item()


def structure_stats(gaussians):
    """End-of-training structure (PLAN_V2 C3): opacity and scale quantiles, neighbour spacing."""
    from simple_knn._C import distCUDA2
    opa = gaussians.get_opacity_.detach().flatten().float()
    smax = gaussians.get_scaling.detach().max(dim=1).values.float()
    knn = torch.sqrt(torch.clamp_min(distCUDA2(gaussians.get_xyz.detach().float()), 0.0))
    q = torch.tensor([0.1, 0.5, 0.9], device=opa.device)
    oq, sq = torch.quantile(opa, q).tolist(), torch.quantile(smax, q).tolist()
    return dict(opacity_q10=round(oq[0], 5), opacity_q50=round(oq[1], 5), opacity_q90=round(oq[2], 5),
                frac_opacity_gt_0_9=round((opa > 0.9).float().mean().item(), 5),
                scale_max_q50=round(sq[1], 6), scale_max_q90=round(sq[2], 6),
                knn_dist_q50=round(torch.median(knn).item(), 6))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, near_range, seed=0):
    assert opt.depth_render in DEPTH_RENDER_MODES, "--depth_render must be one of %s" % (DEPTH_RENDER_MODES,)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    scene_sprical = RenderScene(dataset, gaussians, spiral=True)
    gaussians.training_setup(opt)
    if checkpoint:
        # (model_params, first_iter) = torch.load(checkpoint)
        # gaussians.restore(model_params, opt)
        (model_params, _) = load_checkpoint(checkpoint)
        gaussians.load_shape(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_sprical_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ascii=True, dynamic_ncols=True)
    first_iter += 1

    patch_range = (5, 17) # LLFF

    time_accum = 0

    # Diagnostics (PLAN W4): one diag.csv row every DIAG_EVERY iterations, from a separate RNG
    diag_gen = torch.Generator(device="cuda")
    diag_gen.manual_seed(1000 + seed)
    diag_cam = scene.getTrainCameras()[0]
    diag_file = open(os.path.join(scene.model_path, "diag.csv"), "w")
    diag_file.write(",".join(DIAG_COLUMNS) + "\n")
    diag_file.flush()
    torch.cuda.synchronize()
    t_train0 = t_row = time.time()
    diag_time = 0.0

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(max(iteration - opt.position_lr_start, 0))

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # DropGaussian schedule, verbatim from the official code (0.2 * iteration / 10000); D1: not rescaled to opt.iterations.
        # One mask per iteration: the Gaussian count only changes at the end of the iteration.
        drop_rate_now = opt.drop_rate * iteration / 10000
        drop_comp = None
        if drop_rate_now > 0:
            drop_comp = torch.nn.Dropout(p=drop_rate_now)(torch.ones(gaussians.get_xyz.shape[0], device="cuda"))
        hard_comp = drop_comp if opt.depth_render in ("shared", "hard_only") else None
        soft_comp = drop_comp if opt.depth_render in ("shared", "soft_only") else None

        if not viewpoint_sprical_stack:
            viewpoint_sprical_stack = scene_sprical.getRenderCameras().copy()


        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # -------------------------------------------------- DEPTH --------------------------------------------
        if opt.depth_scale > 0 and iteration > opt.hard_depth_start:
            render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, background, drop_comp=hard_comp)
            depth = render_pkg["depth"]

            # Depth loss
            loss_hard = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 0.1 * loss_l2_dpt

            
            if iteration > 3000:
                loss_hard += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_hard += 1 * loss_global

            (opt.depth_scale * loss_hard).backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        # -------------------------------------------------- pnt --------------------------------------------
        if opt.depth_scale > 0 and iteration > opt.soft_depth_start :
            render_pkg = render_for_opa(viewpoint_cam, gaussians, pipe, background, drop_comp=soft_comp)
            viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]

            # Depth loss
            loss_pnt = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.1 * loss_l2_dpt

            if iteration > 3000:
                loss_pnt += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 1 * loss_global

            (opt.depth_scale * loss_pnt).backward()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        

        # ---------------------------------------------- Photometric --------------------------------------------
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, drop_comp=drop_comp)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # depth
        depth, opacity, alpha = render_pkg["depth"], render_pkg["opacity"], render_pkg['alpha']  # [visibility_filter]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_rgb = loss.detach()


        # Reg
        loss_reg = torch.tensor(0., device=loss.device)
        shape_pena = (gaussians.get_scaling.max(dim=1).values / gaussians.get_scaling.min(dim=1).values).mean()
        scale_pena = ((gaussians.get_scaling.max(dim=1, keepdim=True).values)**2).mean()
        opa_pena = 1 - (opacity[opacity > 0.2]**2).mean() + ((1 - opacity[opacity < 0.2])**2).mean()

        loss_reg += opt.shape_pena*shape_pena + opt.scale_pena*scale_pena + opt.opa_pena*opa_pena
        loss += loss_reg

        loss.backward()
        
        # ================================================================================

        iter_end.record()

        if iteration % DIAG_EVERY == 0:  # before densification, which changes the Gaussian count on the same multiples
            with torch.no_grad():
                torch.cuda.synchronize()
                t_diag0 = time.time()
                iter_ms = (t_diag0 - t_row) * 1000.0 / DIAG_EVERY
                clean = depth_diag(viewpoint_cam, gaussians, pipe, background, iteration, opt, None, None)
                used = clean if hard_comp is None and soft_comp is None else \
                    depth_diag(viewpoint_cam, gaussians, pipe, background, iteration, opt, hard_comp, soft_comp)
                var_depth, var_rgb = mask_variances(diag_cam, gaussians, pipe, background, diag_gen)
                row = [iteration, loss_rgb.item(), used, clean, var_depth, gaussians.get_xyz.shape[0], drop_rate_now,
                       iter_ms, torch.cuda.max_memory_reserved() / 2**20, var_rgb]
                diag_file.write(",".join("%.6g" % v if isinstance(v, float) else str(v) for v in row) + "\n")
                diag_file.flush()
                torch.cuda.synchronize()
                t_row = time.time()  # diagnostics time is excluded from the next row's iter_ms
                diag_time += t_row - t_diag0

        with torch.no_grad():
            # Progress bar
            if not loss.isnan():
                ema_loss_for_log = 0.4 * (loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            clean_iterations = testing_iterations + [first_iter]
            clean_views(iteration, clean_iterations, scene, gaussians, pipe, background)
            time_accum += iter_start.elapsed_time(iter_end)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, render(viewpoint_cam, gaussians, pipe, background)["color"])

            # Densification
            if iteration < opt.densify_until_iter and iteration not in clean_iterations:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:  
                    size_threshold = max_dist = None
                            
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, opt.split_opacity_thresh, max_dist)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            if (iteration - 1) % 25 == 0:
                viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                mask_near = None
                if iteration > 2000:
                    for idx, view in enumerate(scene_sprical.getRenderCameras().copy()):
                        mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near_range
                        mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
                    gaussians.prune_points(mask_near.squeeze())


                ## render process
                # if (iteration + 25) > (opt.iterations):
                #     while viewpoint_sprical_stack:
                #         render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=False)
                #         iteration += 1
                #         viewpoint_sprical_cam = viewpoint_sprical_stack.pop(0)
                # render_one_step(iteration, time_accum / 1000, dataset, viewpoint_sprical_cam, gaussians, render, (pipe, background), save=((iteration + 25) > (opt.iterations)))


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt_latest.pth")

    diag_file.close()
    torch.cuda.synchronize()
    # train_stats.json (PLAN W5 + PLAN_V2 C3)
    stats = dict(
        train_time_s=round(time.time() - t_train0, 1),   # whole loop, incl. the final eval/save and diagnostics
        diag_time_s=round(diag_time, 1),   # C, D, D_hard, D_soft render 2 more diagnostic passes per row than A, B, E
        n_gaussians=int(gaussians.get_xyz.shape[0]),
        vram_peak_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
        model_size_mb=round(sum(p.numel() * p.element_size()
                                for g in gaussians.optimizer.param_groups for p in g["params"]) / 2**20, 2),
        drop_rate=opt.drop_rate, depth_scale=opt.depth_scale, depth_render=opt.depth_render,
        seed=seed, torch_version=torch.__version__)
    with torch.no_grad():
        stats.update(structure_stats(gaussians))
    with open(os.path.join(scene.model_path, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
            

def write_run_config(args):
    """config.json: all arguments + git commit + GPU name (team output layout)."""
    if not args.model_path:
        return
    import json, subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        commit = ""
    os.makedirs(args.model_path, exist_ok=True)
    cfg = dict(vars(args), git_commit=commit, gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(args.model_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def prepare_output_and_logger(args, opt):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "opt_args"), 'w') as opt_log_f:
        opt_log_f.write(str(Namespace(**vars(opt))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def clean_views(iteration, test_iterations, scene, gaussians, pipe, background):
    if iteration in test_iterations:
        visible_pnts = None
        for viewpoint_cam in scene.getTrainCameras().copy():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            visibility_filter = render_pkg["visibility_filter"]
            if visible_pnts is None:
                visible_pnts = visibility_filter
            visible_pnts += visibility_filter
        unvisible_pnts = ~visible_pnts
        gaussians.prune_points(unvisible_pnts)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, depth_loss=torch.tensor(0), reg_loss=torch.tensor(0)):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('train_loss_patches/depth_kl_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/reg_loss', reg_loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'eval', 'cameras' : scene.getEvalCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    depth = render_results["depth"]
                    depth = 1 - (depth - depth.min()) / (depth.max() - depth.min())
                    alpha = render_results["alpha"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bg_mask = (gt_image.max(0, keepdim=True).values < 30/255)
                    bg_mask_clone = bg_mask.clone()
                    for i in range(1, 50):
                        bg_mask[:, i:] *= bg_mask_clone[:, :-i]
                    white_mask = (gt_image.min(0, keepdim=True).values > 240/255)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/mask".format(viewpoint.image_name), bg_mask[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_alpha/white_mask".format(viewpoint.image_name), white_mask[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()



def render_one_step(iteration, time, dataset, viewpoint, gaussians, renderFunc, renderArgs, save=False):
    torch.cuda.empty_cache()
    time_path = os.path.join(dataset.model_path, 'time')
    makedirs(time_path, exist_ok=True)
    render_results = renderFunc(viewpoint, gaussians, *renderArgs)
    image = torch.clamp(render_results["render"], 0.0, 1.0)
    torchvision.utils.save_image(image, os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))
    
    import matplotlib.font_manager as fm # to create font
    from PIL import Image, ImageDraw, ImageFont  
    
    img = Image.open(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))  
    draw = ImageDraw.Draw(img)  
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 30)  
    text = format(time, '.2f') + ' s'
    x = 10
    y = 0 
    draw.text((x-1, y), text, font=font, fill='black')  
    draw.text((x+1, y), text, font=font, fill='black')  
    draw.text((x, y-1), text, font=font, fill='black')  
    draw.text((x, y+1), text, font=font, fill='black')  
    draw.text((x-1, y-1), text, font=font, fill='black')  
    draw.text((x+1, y-1), text, font=font, fill='black')  
    draw.text((x-1, y+1), text, font=font, fill='black')  
    draw.text((x+1, y+1), text, font=font, fill='black')  
    draw.text((x, y), text, font=font, fill='white')

    img.save(os.path.join(time_path, '{0:05d}'.format(iteration) + ".png"))


    torch.cuda.empty_cache()
    if save:
        # os.system(f"ffmpeg -i " + time_path + f"/%5d.png -q 2 " + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))
        os.system(f'ffmpeg -f image2 -pattern_type glob -i "' + time_path + f'/*.png" -q 2 ' + dataset.model_path + "/out_time_{}.mp4 -y".format(dataset.model_path.split('/')[-1]))





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[6000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--near", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)
    write_run_config(args)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.near, args.seed)

    # All done
    print("\nTraining complete.")
