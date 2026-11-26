#!/bin/bash
# Double-click to start GastroMap. Your browser opens automatically.
# Close this window (or press Ctrl+C) to stop the tool.
cd "$(dirname "$0")"
clear
echo "========================================"
echo "  GastroMap - starting, please wait..."
echo "  (first start takes ~1 minute to load the model)"
echo "========================================"
if [ ! -f "$HOME/gcx-env/bin/activate" ]; then
  echo "Setup not found. Please double-click 'Install GastroMap.command' first."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi
source "$HOME/gcx-env/bin/activate"
python code/tool/app.py
echo; read -n 1 -s -r -p "GastroMap stopped. Press any key to close this window."
