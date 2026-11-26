"""Index both datasets into CSV tables and attach slide/group proxies.

Usage:
    python scripts/01_build_index.py --gashis_root /data/GasHisSDB --gashis_subset 160 \
        --hmu_root /data/HMU-GC-HE-30K --out data/index
Optional:
    --hmu_mapping tum_vs_rest      (sensitivity analysis; default tum_vs_nor)
    --hmu_block 13                 (tiles per slide block)
    --gashis_pseudogroups 600      (cluster GasHisSDB into ~600 source-image proxies; slow-ish)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcx.data import (index_gashissdb, index_hmu, assign_contiguous_groups,  # noqa: E402
                      assign_pseudo_groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gashis_root")
    ap.add_argument("--gashis_subset", default="160")
    ap.add_argument("--gashis_pseudogroups", type=int, default=0)
    ap.add_argument("--hmu_root")
    ap.add_argument("--hmu_mapping", default="tum_vs_nor", choices=["tum_vs_nor", "tum_vs_rest"])
    ap.add_argument("--hmu_block", type=int, default=13)
    ap.add_argument("--out", default="data/index")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.gashis_root:
        g = index_gashissdb(a.gashis_root, a.gashis_subset)
        if a.gashis_pseudogroups:
            g = assign_pseudo_groups(g, a.gashis_pseudogroups)
        g.to_csv(out / "gashissdb.csv", index=False)
        print(f"GasHisSDB: {len(g)} patches | per class:\n{g.label.value_counts().to_string()}")

    if a.hmu_root:
        h = index_hmu(a.hmu_root, a.hmu_mapping)
        h = assign_contiguous_groups(h, a.hmu_block)
        h_all = h.copy()
        h = h[h.label >= 0]
        h.to_csv(out / "hmu.csv", index=False)
        h_all.to_csv(out / "hmu_all_classes.csv", index=False)
        print(f"HMU ({a.hmu_mapping}): {len(h)} patches, {h.group.nunique()} slide-blocks | per class:\n"
              f"{h.orig_class.value_counts().to_string()}")


if __name__ == "__main__":
    main()
