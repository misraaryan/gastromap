"""Copy finished HiESD downloads (checksum-verified) from ~/Downloads into data/raw/HiESD/slides and tile them."""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; H = ROOT / "data/raw/HiESD"
DL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.path.expanduser("~/mnt/Downloads"))
sel = pd.read_csv(H / "meta/selected_14_slides.csv")
for r in sel.itertuples():
    src, dst = DL / f"{r.svs}.svs", H / "slides" / f"{r.svs}.svs"
    status = "-"
    if dst.exists() and dst.stat().st_size == r.bytes:
        status = "in project"
    elif src.exists() and src.stat().st_size == r.bytes:
        h = hashlib.md5(open(src, "rb").read()).hexdigest()
        if h == r.md5:
            shutil.copyfile(src, dst); status = "copied (md5 ok)"
        else:
            status = f"MD5 MISMATCH {h}"
    elif src.exists():
        status = f"downloading {src.stat().st_size / 1e6:.0f}/{r.bytes / 1e6:.0f} MB"
    tiles = H / "tiles" / f"{r.slide}.csv"
    if dst.exists() and dst.stat().st_size == r.bytes and not tiles.exists():
        out = subprocess.run([sys.executable, str(ROOT / "code/scripts/16_hiesd_tiles.py"), "--slide", r.slide],
                             capture_output=True, text=True)
        status += " | " + (out.stdout.strip().splitlines() or [out.stderr.strip()[-200:]])[-1]
    elif tiles.exists():
        status += " | tiled"
    print(f"{r.slide:9s} {status}")
