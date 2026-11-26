#!/bin/bash
# One-time setup for GastroMap (about 5 minutes). Double-click to run.
cd "$(dirname "$0")"
clear
echo "Installing GastroMap (one time only)..."
if [ ! -f "$HOME/gcx-env/bin/activate" ]; then
  python3 -m venv "$HOME/gcx-env" || { echo "Python 3 is needed: install it from python.org"; read -n 1; exit 1; }
fi
source "$HOME/gcx-env/bin/activate"
pip install --upgrade pip -q
pip install -q torch torchvision timm transformers huggingface_hub scikit-learn pandas pillow matplotlib gradio openslide-python openslide-bin
python -c "import torch, gradio, openslide; print('All good. Apple GPU available:', torch.backends.mps.is_available())"
echo
echo "The UNI model needs a one-time Hugging Face login (approved access to MahmoodLab/UNI)."
echo "If you are not logged in yet, a login prompt follows. Choose 'Log in with your browser'."
hf auth whoami >/dev/null 2>&1 || hf auth login
echo
read -n 1 -s -r -p "Setup finished. Press any key to close. Then double-click 'GastroMap.command'."
