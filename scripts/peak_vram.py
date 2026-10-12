#!/usr/bin/env python
"""Run a Python script in-process and report its peak GPU memory.

Usage: python scripts/peak_vram.py train_llff.py -s data/llff/fern --model_path ... <flags>

WSL's nvidia-smi cannot show per-process memory, so this reads PyTorch's own
counters (exact, this process only) and, as a cross-check, polls nvidia-smi for
the whole-GPU total minus what was in use before the script started.
"""
import os, runpy, subprocess, sys, threading, time
import torch


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return int(out[0]), int(out[1])


baseline, total = gpu_used_mb()
peak = [baseline]
done = threading.Event()


def poll():
    while not done.is_set():
        peak[0] = max(peak[0], gpu_used_mb()[0])
        time.sleep(0.5)


threading.Thread(target=poll, daemon=True).start()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(os.path.abspath(script))  # imports resolve as if run directly
try:
    runpy.run_path(script, run_name="__main__")
finally:
    done.set()
    reserved = torch.cuda.max_memory_reserved() / 2**20
    allocated = torch.cuda.max_memory_allocated() / 2**20
    msg = (f"\n==== PEAK VRAM ({script}) ====\n"
           f"torch max_memory_reserved : {reserved:7.0f} MB   <- use this as vram_peak_mb\n"
           f"torch max_memory_allocated: {allocated:7.0f} MB\n"
           f"nvidia-smi whole GPU      : {peak[0]:7.0f} MB peak, {baseline} MB before start, "
           f"delta {peak[0] - baseline} MB, card total {total} MB\n"
           f"Real footprint on the card = the nvidia-smi delta (tensors + CUDA context + kernels).\n"
           f"Laptop OK for full runs if peak stays under about 3600 MB.\n")
    sys.__stdout__.write(msg)
    sys.__stdout__.flush()
