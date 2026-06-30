"""Regenerate the bundled chassis-sample parquet at data/m5/sample_smooth.parquet.

One-shot deterministic script. Run from the repo root:

    uv run python scripts/generate_m5_sample.py

The output is a 1825-row parquet (5 years daily) shaped after the
"fast-moving smooth" archetype: ``demand`` drawn IID from ``Normal(10, 2)``
clipped at 0 with ``master_seed=42``. **The "M5" label in the directory
name is aspirational for the chassis** — real Walmart M5
SKUs ship via the archetype curation, which swaps in
parquets sourced from the actual Makridakis M5 competition data. The
synthetic chassis sample exists so the empirical demand chassis is
exercisable end-to-end with hermetic tests (no network, no external
dataset, no legal redistribution questions) and the resulting policy
outcome is verifiable against the established Normal(10, 2) cost
baseline of $5,562.05.

Re-running this script must produce a byte-identical parquet —
deterministic seed (42) + deterministic numpy ops + parquet's stable
encoding for the same dtype/column layout. If the file size or contents
ever change after a re-run, investigate (likely a numpy/pyarrow
behavioral shift) rather than committing the drifted file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "m5" / "sample_smooth.parquet"

N_PERIODS = 1825  # 5 years daily
MEAN = 10.0
STD = 2.0
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    history = np.maximum(0.0, rng.normal(MEAN, STD, size=N_PERIODS))
    assert np.isfinite(history).all()
    assert (history >= 0.0).all()
    df = pd.DataFrame({"demand": history})
    df.to_parquet(OUTPUT, index=False)
    size = OUTPUT.stat().st_size
    print(
        f"Wrote {N_PERIODS} rows to {OUTPUT.relative_to(REPO_ROOT)} "
        f"({size} bytes; mean={history.mean():.4f}, std={history.std():.4f}, "
        f"min={history.min():.4f}, max={history.max():.4f})"
    )


if __name__ == "__main__":
    main()
