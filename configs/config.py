"""
configs/config.py
=================
Central configuration for the polypharmacy side effect prediction pipeline.
Edit DATA_DIR to point to your local copy of the Decagon dataset files.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get("DECAGON_DATA", "data/raw")
OUTPUT_DIR = os.environ.get("DECAGON_OUT",  "results")

RAW_FILES = {
    "combo"       : os.path.join(DATA_DIR, "bio-decagon-combo.csv"),
    "mono"        : os.path.join(DATA_DIR, "bio-decagon-mono.csv"),
    "ppi"         : os.path.join(DATA_DIR, "bio-decagon-ppi.csv"),
    "targets"     : os.path.join(DATA_DIR, "bio-decagon-targets.csv"),
}

CACHE_DIR  = os.path.join(OUTPUT_DIR, "cache")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# ── Dataset filters ────────────────────────────────────────────────────────────
MIN_PAIRS_PER_SE = 500          # keep only SEs with >= this many drug pairs
RANDOM_SEED      = 42

# ── The 12 representative SEs for detailed analysis (UMLS CUI → readable name)
REPRESENTATIVE_SES = {
    # high pairs
    "C0034065": "Pulmonary embolism",
    "C0020473": "Hyperlipaemia",
    "C1510472": "Drug addiction",
    # medium pairs
    "C0001824": "Agranulocytosis",
    "C0019348": "Herpes simplex",
    "C0085606": "Micturition urgency",
    # low pairs
    "C0025289": "Meningitis",
    "C1510431": "Superficial thrombophlebitis",
    "C0029878": "External ear infection",
    # very low pairs
    "C0243010": "Viral encephalitis",
    "C0042164": "Intraocular inflammation",
    "C0332687": "Burns second degree",
}

# ── Splitting ──────────────────────────────────────────────────────────────────
SPLIT_STRATEGIES  = ["random_pair", "stratified_pair", "drug_cold_start"]
DEFAULT_SPLIT     = "drug_cold_start"
TRAIN_RATIO       = 0.70
VAL_RATIO         = 0.10
TEST_RATIO        = 0.20
COLD_START_DRUG_FRAC = 0.20   # fraction of drugs held out entirely in cold-start

# ── Negative sampling ──────────────────────────────────────────────────────────
NEG_STRATEGIES = ["random", "structured"]   # structured = cross-SE negatives
NEG_RATIO      = 1.0                        # negatives per positive

# ── Feature settings ───────────────────────────────────────────────────────────
MORGAN_RADIUS = 2
MORGAN_NBITS  = 2048
PHYSCHEM_FEATURES = True        # append RDKit physicochemical descriptors
PPI_HOPS      = 2               # neighbourhood hops for protein propagation

# ── Drug pair operators ────────────────────────────────────────────────────────
PAIR_OPERATORS = ["hadamard", "absdiff", "hadamard_absdiff", "concat", "sum"]
DEFAULT_PAIR_OP = "hadamard_absdiff"

# ── Model hyperparameters ──────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators"   : 500,
    "learning_rate"  : 0.05,
    "num_leaves"     : 63,
    "min_child_samples": 20,
    "subsample"      : 0.8,
    "colsample_bytree": 0.8,
    "class_weight"   : "balanced",
    "random_state"   : RANDOM_SEED,
    "n_jobs"         : -1,
    "verbose"        : -1,
}

XGB_PARAMS = {
    "n_estimators"   : 500,
    "learning_rate"  : 0.05,
    "max_depth"      : 6,
    "subsample"      : 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,      # set per-SE based on class ratio
    "eval_metric"    : "auc",
    "random_state"   : RANDOM_SEED,
    "n_jobs"         : -1,
    "verbosity"      : 0,
}

CATBOOST_PARAMS = {
    "iterations"     : 500,
    "learning_rate"  : 0.05,
    "depth"          : 6,
    "random_seed"    : RANDOM_SEED,
    "verbose"        : 0,
    "thread_count"   : -1,
}

RGCN_PARAMS = {
    "hidden_dim"     : 64,
    "num_bases"      : 32,       # basis decomposition across 963 relations
    "num_layers"     : 2,
    "dropout"        : 0.3,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-4,
    "epochs"         : 100,
    "batch_size"     : 512,      # number of edges per mini-batch (GraphSAINT-style)
    "random_state"   : RANDOM_SEED,
}

# ── Evaluation ─────────────────────────────────────────────────────────────────
METRICS = ["auroc", "ap", "f1", "precision", "recall"]
