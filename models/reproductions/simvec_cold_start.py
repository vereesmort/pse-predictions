"""
models/reproductions/simvec_cold_start.py
==========================================
SimVec reproduction under full drug cold-start split.

Convenience entry-point that calls simvec_weak_node.py with
--split drug_cold_start. See simvec_weak_node.py for full documentation.

Usage
-----
    python models/reproductions/simvec_cold_start.py
    python models/reproductions/simvec_cold_start.py --run-both
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import pandas as pd
from models.reproductions.simvec_weak_node import run_simvec
from models.reproductions._utils import compare_protocols, save_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses",          default="rep")
    parser.add_argument("--dim",          type=int, default=100)
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--no-chem-init", action="store_true")
    parser.add_argument("--no-sim-edges", action="store_true")
    parser.add_argument("--run-both",     action="store_true",
                        help="Run weak_node + cold_start side by side")
    args = parser.parse_args()
    se_ids = None if args.ses == "rep" else "all"

    if args.run_both:
        r_wn = run_simvec(se_ids, "weak_node",       args.dim, args.epochs)
        r_cs = run_simvec(se_ids, "drug_cold_start", args.dim, args.epochs,
                          use_chem_init=not args.no_chem_init,
                          use_sim_edges=not args.no_sim_edges)
        r_wn["protocol"] = "original (weak-node)"
        r_cs["protocol"] = "fair (drug cold-start)"
        combined = pd.concat([r_wn, r_cs], ignore_index=True)
        compare_protocols(combined, "SimVec")
        save_results(combined, "simvec_combined")
    else:
        run_simvec(se_ids=se_ids, split_type="drug_cold_start",
                   dim=args.dim, epochs=args.epochs,
                   use_chem_init=not args.no_chem_init,
                   use_sim_edges=not args.no_sim_edges)
