"""Run the independent FixedV1 on Person 2's v1 golden vectors and compare bit for bit.
    .venv/bin/python verify_vectors.py [vector_dir ...]    (default: v1_contract and v1_contract/edge)
"""
import sys
from pathlib import Path
import numpy as np
from evm_v1 import FixedV1, coeff_q016, alpha_q88, load_vectors, FC_HI, FC_LO, ALPHA

ROOT = Path(__file__).resolve().parents[2] / "test_vectors" / "v1_contract"


def check(d: Path) -> bool:
    p, fin, fout, st = load_vectors(d)
    ok = True
    if "fc_hi" in p:  # coefficients are derivable only for the real-clip set
        mine = (coeff_q016(p["fc_hi"], p["fs"]), coeff_q016(p["fc_lo"], p["fs"]), alpha_q88(p["alpha_float"]))
        theirs = (p["r_hi"], p["r_lo"], p["alpha"])
        print(f"  coefficients mine {mine} theirs {theirs} -> {'match' if mine == theirs else 'DIFFER'}")
        ok &= mine == theirs
    m = FixedV1(p["r_hi"], p["r_lo"], p["alpha"])
    bad = 0
    for n in range(p["N"]):
        out = m.process(fin[n], init=n in p["init_frames"])
        if n in p["init_frames"] and out.any():
            print(f"  frame {n}: INIT frame output not all zero"); ok = False
        diff = np.argwhere(out != fout[n])
        if len(diff):
            bad += len(diff)
            if bad <= 10:
                y, x, c = diff[0]
                print(f"  MISMATCH frame {n} pixel (y={y},x={x},ch={'BGR'[c]}): mine {out[y,x,c]} expected {fout[n][y,x,c]}")
            ok = False
        if n < p["state_frames"]:
            lh, ll = m.state_int32()
            if not (np.array_equal(lh, st[n, 0]) and np.array_equal(ll, st[n, 1])):
                print(f"  STATE mismatch after frame {n}"); ok = False
    print(f"  {d.relative_to(ROOT.parent)}: {p['N']} frames, {bad} output mismatches -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT, ROOT / "edge"]
    sys.exit(0 if all([check(d) for d in dirs]) else 1)
