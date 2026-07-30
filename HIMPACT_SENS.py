#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIMPACT_SENS.py
===========================================================================
Sensitivity runner for HIMPACT_v1_6.py.
===========================================================================
Parallelises all combinations of:
  SEARCH_RADIUS_KM      : [50, 100, 150]
  INTERP_LEVELS_HPA     : {None, [800,850,900,950], arange(500,1025,25)}
  PERCENTILE_THRESHOLD  : [1, 3, 5]
  TRACK_CENTROID / TRACK_MINIMUM (valid modes):
        Ct_Mt  → all percentile values
        Ct_Mf  → all percentile values
        Cf_Mt  → PERCENTILE_THRESHOLD = 1 only   (minimum-only, no centroid)
        Cf_Mf  → EXCLUDED (both False is invalid)

Workers = (number of physical CPU cores) − 1.

Each combination is launched as an independent subprocess of HIMPACT_v1_6.py,
with parameters injected via a temporary JSON file read by Section 1b of
that script (HIMPACT_SENS_JSON env var).  Every subprocess captures its
stdout/stderr to a per-combination .log file.

Output naming per combination
─────────────────────────────
  sim label  :  <n_levels>v_<radius>km_<pct>p_C<t/f>_M<t/f>
                  n_levels = len(INTERP_LEVELS_HPA) + 1   (the +1 is SLP)
                  example  : 5v_150km_5p_Ct_Mf
  outfolder  :  {base_outfolder}/HIMPACT_SENS_{sim_label}

Fixed parameters (CYCLONE, START_DATE, S0LAT, S0LON, SMOOTHING_WINDOW,
LANDFALL_DETECTION, EXPORT_VARIABLES, PLOT, DO_EXPORT_VARIABLES,
SAVE_ALL_TRACKS, CHECK_PLOTS) are defined ONCE in HIMPACT_v1_6.py
Section 1 and are NOT overridden here.
===========================================================================
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import concurrent.futures

import numpy as np


# =============================================================================
# 0.  PHYSICAL CORE DETECTION
# =============================================================================

def _detect_physical_cores() -> int:
    """Return the number of physical (non-HT) CPU cores."""
    # Preferred: psutil (often available in scientific Python stacks)
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n and n > 0:
            return n
    except ImportError:
        pass

    # Fallback: parse lscpu (standard on Linux/HPC)
    try:
        res = subprocess.run(
            "lscpu | awk -F: "
            "'/^Core\\(s\\) per socket/{gsub(/ /,\"\",$2); c=$2} "
            "/^Socket\\(s\\)/{gsub(/ /,\"\",$2); s=$2} "
            "END{printf \"%d\", c+0, s+0; printf \"%d\", (c+0)*(s+0)}'",
            shell=True, capture_output=True, text=True,
        )
        val = res.stdout.strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    except Exception:
        pass

    # Last resort: half of logical count (typical HT ratio)
    return max(1, (os.cpu_count() or 2) // 2)


_PHYSICAL_CORES = _detect_physical_cores()
N_WORKERS       = max(1, _PHYSICAL_CORES - 1)


# =============================================================================
# 1.  BASE MODELS CONFIGURATION
#     ─── Only model / infolder / outfolder go here. ───────────────────────
#     CYCLONE, START_DATE, S0LAT, S0LON and all other fixed tracking params
#     are defined once in HIMPACT_v1_6.py Section 1; do NOT duplicate them.
# =============================================================================

MODELS_CONFIG = [
#   ─────────────────────────────────────────────────────────────────────────
    # Example MPAS configuration:
    # {
    #     "model":     "MPAS",
    #     "infolder":  "/path/to/your/MPAS/data",
    #     "outfolder": "/path/to/your/output/HIMPACT_SENS",
    # },
    # Example ICON configuration:
    # {
    #     "model":     "ICON",
    #     "infolder":  "/path/to/your/ICON/data",
    #     "outfolder": "/path/to/your/output/HIMPACT_SENS",
    # },
    # Example WRF configuration:
    # {
    #     "model":     "WRF",
    #     "infolder":  "/path/to/your/WRF/data",
    #     "outfolder": "/path/to/your/output/HIMPACT_SENS",
    # },
    # Example ERA5 configuration:
    # {
    #     "model":     "ERA5",
    #     "infolder":  "/path/to/your/ERA5/data",
    #     "outfolder": "/path/to/your/output/HIMPACT_SENS",
    # },
#   ─────────────────────────────────────────────────────────────────────────
]


# =============================================================================
# 2.  SENSITIVITY PARAMETER SPACE
# =============================================================================

SEARCH_RADIUS_KM_LIST = [50, 100]

# Each entry is None (SLP-only) or a plain Python list[int] (JSON-serialisable).
# The sim-label prefix digit = len(entry) + 1  (the +1 counts SLP).
INTERP_LEVELS_HPA_LIST = [
#    None,                                          # SLP only  → prefix "1v"
#    [800, 850, 900, 950],                          # 4 levels  → prefix "5v"
    np.arange(500, 1025, 25, dtype=int).tolist(),  # 21 levels → prefix "22v"
]

PERCENTILE_THRESHOLD_LIST = [1, 3, 5]

# Valid tracking mode combinations: (TRACK_CENTROID, TRACK_MINIMUM, short_label)
#   Cf_Mf → excluded (both False crashes HIMPACT)
#   Cf_Mt → PERCENTILE_THRESHOLD is irrelevant for minimum-only tracking;
#            use only pct=1 to avoid identical redundant runs.
TRACKING_MODES = [
    (True,  True,  "Ct_Mt"),   # centroid + minimum — all percentiles
    (True,  False, "Ct_Mf"),   # centroid only      — all percentiles
    (False, True,  "Cf_Mt"),   # minimum only       — pct = 1 only
]


# =============================================================================
# 3.  HELPERS
# =============================================================================

def make_sim_label(levels, radius, pct, centroid, minimum) -> str:
    """
    Build the sensitivity simulation label.

    Format: <n_levels>v_<radius>km_<pct>p_C<t/f>_M<t/f>
      n_levels = len(INTERP_LEVELS_HPA) + 1   (+1 for SLP always included)

    Example: levels=[800,850,900,950], radius=150, pct=5, Ct, Mf → '5v_150km_5p_Ct_Mf'
    """
    n = (0 if levels is None else len(levels)) + 1
    c = "t" if centroid else "f"
    m = "t" if minimum else "f"
    return f"{n}v_{radius}km_{pct}p_C{c}_M{m}"


def build_combinations(model_cfg: dict) -> list[dict]:
    """
    Enumerate all valid parameter combinations for *model_cfg* and return
    a list of JSON-serialisable override dicts (one per subprocess call).
    """
    combos = []
    for centroid, minimum, _label in TRACKING_MODES:
        # Cf_Mt: percentile has no effect on minimum-based centroid isolation;
        # collapse to pct=1 to avoid producing duplicate outputs.
        pct_values = [1] if not centroid else PERCENTILE_THRESHOLD_LIST

        for radius in SEARCH_RADIUS_KM_LIST:
            for levels in INTERP_LEVELS_HPA_LIST:
                for pct in pct_values:
                    sim = make_sim_label(levels, radius, pct, centroid, minimum)
                    out = os.path.join(model_cfg["outfolder"], f"HIMPACT_SENS_{sim}")
                    combos.append({
                        "SEARCH_RADIUS_KM":     radius,
                        "INTERP_LEVELS_HPA":    levels,    # None  or  list[int]
                        "PERCENTILE_THRESHOLD": pct,
                        "TRACK_CENTROID":       centroid,
                        "TRACK_MINIMUM":        minimum,
                        "MODELS_CONFIG": {
                            "model":     model_cfg["model"],
                            "sim":       sim,
                            "infolder":  model_cfg["infolder"],
                            "outfolder": out,
                        },
                    })
    return combos


# =============================================================================
# 4.  WORKER  (runs one combination as an isolated subprocess)
# =============================================================================

# HIMPACT_v1_6.py must live in the same directory as this script.
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
HIMPACT_SCRIPT = os.path.join(_SCRIPT_DIR, "HIMPACT_v1_6.py")


def run_one(combo: dict) -> tuple:
    """
    Write *combo* to a temp JSON, set HIMPACT_SENS_JSON in the child
    environment, and block until the HIMPACT subprocess finishes.

    Returns
    -------
    (sim_label : str,
     returncode: int,   0 = success
     elapsed_s : float,
     log_path  : str)   path to the captured output log
    """
    sim_label = combo["MODELS_CONFIG"]["sim"]
    outfolder  = combo["MODELS_CONFIG"]["outfolder"]

    # ── write override JSON to a named temp file ────────────────────────────
    try:
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
            prefix=f"himpact_sens_{sim_label}_",
        )
        json.dump(combo, tf)
        tf.close()
        json_path = tf.name
    except Exception as exc:
        return (sim_label, -1, 0.0, f"JSON write error: {exc}")

    # ── ensure outfolder exists so the log can always be written ────────────
    os.makedirs(outfolder, exist_ok=True)
    log_path = os.path.join(outfolder, f"himpact_{sim_label}.log")

    # ── launch subprocess ───────────────────────────────────────────────────
    env                    = os.environ.copy()
    env["HIMPACT_SENS_JSON"] = json_path
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, HIMPACT_SCRIPT],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → single stream
            text=True,
        )
        elapsed = time.perf_counter() - t0
        with open(log_path, "w") as lf:
            lf.write(proc.stdout or "")
        return (sim_label, proc.returncode, elapsed, log_path)

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        msg = f"Subprocess error: {exc}"
        try:
            with open(log_path, "w") as lf:
                lf.write(msg)
        except OSError:
            pass
        return (sim_label, -1, elapsed, log_path)

    finally:
        try:
            os.unlink(json_path)   # delete temp JSON regardless of outcome
        except OSError:
            pass


# =============================================================================
# 5.  MAIN
# =============================================================================

if __name__ == "__main__":

    # ── sanity check ─────────────────────────────────────────────────────────
    if not os.path.isfile(HIMPACT_SCRIPT):
        print(f"[ERROR] HIMPACT script not found at: {HIMPACT_SCRIPT}")
        sys.exit(1)

    # ── build full combination list ───────────────────────────────────────────
    all_combos: list[dict] = []
    for cfg in MODELS_CONFIG:
        all_combos.extend(build_combinations(cfg))

    total = len(all_combos)

    # ── print plan ────────────────────────────────────────────────────────────
    SEP = "═" * 72
    print(f"\n{SEP}")
    print("  HIMPACT Sensitivity Runner")
    print(SEP)
    print(f"  Physical cores  : {_PHYSICAL_CORES}")
    print(f"  Workers (N−1)   : {N_WORKERS}")
    print(f"  Models          : {len(MODELS_CONFIG)}")
    print(f"  Combinations    : {total}")
    print(SEP)
    preview_n = min(5, total)
    for c in all_combos[:preview_n]:
        print(f"    {c['MODELS_CONFIG']['sim']}")
    if total > preview_n * 2:
        print(f"    ... ({total - preview_n * 2} more) ...")
    for c in all_combos[-preview_n:]:
        print(f"    {c['MODELS_CONFIG']['sim']}")
    print(SEP + "\n")

    # ── parallel execution (ThreadPool: each thread blocks on a subprocess) ──
    # ThreadPoolExecutor is preferred over ProcessPoolExecutor here because
    # the work is subprocess I/O-bound, not CPU-bound in the runner itself.
    # This avoids an extra layer of forked processes and is simpler to debug.
    completed = 0
    failed: list[tuple] = []
    t_global  = time.perf_counter()

    C_OK   = "\033[32m"
    C_FAIL = "\033[31m"
    C_HDR  = "\033[1;36m"
    C_RST  = "\033[0m"

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, c): c for c in all_combos}

        for fut in concurrent.futures.as_completed(futures):
            sim_label, rc, elapsed, log_or_err = fut.result()
            completed += 1
            if rc == 0:
                status = f"{C_OK}OK{C_RST}"
            else:
                status = f"{C_FAIL}FAIL rc={rc}{C_RST}"
                failed.append((sim_label, rc, log_or_err))

            print(
                f"  [{completed:3d}/{total}] "
                f"{C_HDR}{sim_label:<30}{C_RST} "
                f"{status}  ({elapsed:6.0f}s)"
                f"  →  {log_or_err}"
            )

    # ── summary ───────────────────────────────────────────────────────────────
    wall = time.perf_counter() - t_global
    print(f"\n{SEP}")
    print(f"  Completed {total} combinations in {wall:.0f}s ({wall/60:.1f} min)")
    if failed:
        print(f"\n  {C_FAIL}FAILED ({len(failed)}):{C_RST}")
        for lbl, rc, log in failed:
            print(f"    {lbl:<30}  rc={rc}  log → {log}")
    else:
        print(f"  {C_OK}All combinations completed successfully.{C_RST}")
    print(SEP + "\n")