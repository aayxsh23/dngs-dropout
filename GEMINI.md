@./AGENTS.md

# Gemini specifics

Everything in AGENTS.md applies. Gemini CLI loads this file and expands the `@./AGENTS.md` import above. If your Gemini tool doesn't support imports, read AGENTS.md yourself before doing anything.

- **The Gemini assistant inside Colab does not read this repo's files.** When you ask it about the notebook, paste in the "Environments", "Invariants" and "Practical gotchas" sections of AGENTS.md. In particular, it must never write `torch.load(...)` directly; use `load_checkpoint()`. It must not regenerate depth maps, and it must not change `metrics.json` keys.
- On Colab the repo lives at `/content/dngs-dropout`, and the shared Drive folder at `/content/drive/MyDrive/sparse3dgs`. That path needs the "Add shortcut to My Drive" step from the playbook.
- Colab runs Python 3.13 and PyTorch 2.11, but the same code must also run on the laptop's Python 3.7 / PyTorch 1.12. Don't introduce syntax or APIs newer than Python 3.7 in training code.
