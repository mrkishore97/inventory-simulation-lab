"""Curate one real Walmart M5 SKU per demand archetype into bundled parquets.

One-shot, deterministic curation tooling. Run from the repo
root **after** placing the raw M5 competition CSVs in ``data/m5/raw/``:

    uv run python scripts/extract_m5_archetypes.py

Required inputs (gitignored; NOT committed — the full dataset is ~120 MB):

    data/m5/raw/sales_train_validation.csv   # 30,490 series x d_1..d_N unit sales
    data/m5/raw/calendar.csv                 # d_* <-> wm_yr_wk mapping
    data/m5/raw/sell_prices.csv              # (store, item, wm_yr_wk) -> sell_price

Outputs (committed — tiny, one row-per-day per SKU):

    data/m5/archetype_smooth.parquet         # fast-moving smooth
    data/m5/archetype_intermittent.parquet   # slow-moving intermittent
    data/m5/archetype_seasonal.parquet       # highly seasonal
    data/m5/archetype_promotional.parquet    # promotional-driven
    data/m5/archetypes_manifest.json         # provenance + Syntetos-Boylan labels

Selection is automated and deterministic. The script computes, for every
series, its average demand interval (ADI), squared coefficient of variation of
non-zero demand sizes (CV2), a weekly/yearly seasonal-strength proxy, and a
price-discount promo-intensity, then picks the most representative SKU per
archetype (excluding already-picked SKUs so the four are distinct). The
Syntetos-Boylan-Croston (2005) cutoffs ADI = 1.32 and CV2 = 0.49 partition
series into smooth / erratic / intermittent / lumpy quadrants; each chosen
SKU's actual quadrant is recorded in the manifest so the four
archetype names carry an honest SB label for the SB classifier in
``recommender.classify``.

**IID-resampling caveat (load-bearing).** The bundled parquets feed the generic
``EmpiricalDemand`` arm, whose ``draw()`` returns ``rng.choice(history)`` — a
uniform IID draw. This reproduces each SKU's *marginal* demand distribution
(its zero-fraction, tail, mean) but NOT its *temporal* structure (seasonality,
promo timing, autocorrelation are all lost). A "seasonal" or "promotional"
archetype therefore contributes its fat-tailed marginal, not an in-time wave or
spike train. In-time patterns are the orthogonal Pattern axis
(``SeasonalPattern`` / ``TrendingPattern``); the two axes compose but this
script ships the marginal-replay archetypes only.

Leading zeros (the pre-introduction period before a SKU was first stocked) are
trimmed before metrics and before writing, so the stored history is the SKU's
active-life demand rather than an artificial zero-inflated tail.

Determinism: given the same input CSVs, re-running produces byte-identical
parquets + manifest (fixed cutoffs, deterministic sort with lexicographic
tie-breaks, stable parquet encoding). If outputs drift after a re-run on the
same inputs, investigate rather than committing the drift.

Memory note: the script materialises the d_* matrix as int32 (~230 MB for the
full M5) and a weekly price pivot (~70 MB). Both fit comfortably in a few
hundred MB; if a future, larger dataset strains memory, chunk the per-series
loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import classify

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "m5" / "raw"
OUT_DIR = REPO_ROOT / "data" / "m5"

# Eligibility floors: a candidate series must have at least this many non-zero
# demand periods (clearly "alive", not near-dead) and this many active
# (post-introduction) periods (long enough that yearly autocorrelation is
# meaningful and the history is representative). Guards against degenerate
# CV2 / ADI / seasonal-strength / promo metrics on short or sparse series — the
# permissive 12 / 28 floors let 2-point autocorrelations and near-dead SKUs win.
MIN_NONZERO = 50
MIN_ACTIVE = 730

# Promo detection: a week is "discounted" when its sell price is at least this
# fraction below the SKU's median price.
DISCOUNT_THRESHOLD = 0.90

# Archetype labels, in selection order (each pick excludes SKUs
# already chosen by an earlier archetype so the four are distinct).
ARCHETYPES = ("smooth", "intermittent", "seasonal", "promotional")


# --------------------------------------------------------------------------- #
# Selection-signal helpers (no I/O; unit-tested in tests/unit/test_extract_m5_archetypes.py).
# The SB-classification metrics (ADI / CV² / quadrant / leading-zero trim) now live in
# recommender.classify and are called module-qualified (classify.adi / .cv_squared / ...).
# --------------------------------------------------------------------------- #
def _autocorr(series: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation of ``series`` at ``lag`` (0.0 if undefined).

    Requires at least two full periods (``len >= 2 * lag``) so a short tail
    cannot fake a signal — a 2-point correlation is spuriously +/-1.
    """
    if lag <= 0 or series.shape[0] < 2 * lag:
        return 0.0
    a = series[:-lag]
    b = series[lag:]
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    if std_a == 0.0 or std_b == 0.0:
        return 0.0
    cov = float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    return cov / (std_a * std_b)


def _seasonal_strength(series: np.ndarray) -> float:
    """Seasonal-strength proxy: max autocorrelation at weekly / yearly lag."""
    return max(_autocorr(series, 7), _autocorr(series, 364))


def _promo_intensity(demand: np.ndarray, price: np.ndarray) -> float:
    """Discount frequency x demand lift on discounted vs full-price days.

    ``price`` is the daily sell price aligned to ``demand`` (NaN where unknown).
    Returns 0.0 when prices are unknown or no discounts occur.
    """
    valid = np.isfinite(price)
    if int(np.count_nonzero(valid)) < 2:
        return 0.0
    dem = demand[valid]
    prc = price[valid]
    reference = float(np.median(prc))
    if reference <= 0.0:
        return 0.0
    discounted = prc < (DISCOUNT_THRESHOLD * reference)
    n_discounted = int(np.count_nonzero(discounted))
    if n_discounted == 0:
        return 0.0
    frequency = float(n_discounted) / float(prc.shape[0])
    mean_on = float(np.mean(dem[discounted]))
    full_price = dem[~discounted]
    mean_off = float(np.mean(full_price)) if full_price.shape[0] > 0 else 0.0
    if mean_off <= 0.0:
        lift = 1.0 if mean_on > 0.0 else 0.0
    else:
        lift = max(0.0, mean_on / mean_off - 1.0)
    return frequency * lift


# --------------------------------------------------------------------------- #
# Data loading + alignment (I/O; validated against real CSVs in Phase 2)
# --------------------------------------------------------------------------- #
def _day_columns(columns: list[str]) -> list[str]:
    """Return the d_* columns sorted by integer day index."""
    day_cols = [c for c in columns if c.startswith("d_")]
    return sorted(day_cols, key=lambda c: int(c.split("_", 1)[1]))


def _load_sales(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load sales; return (metadata frame, day columns, int32 demand matrix)."""
    header = pd.read_csv(path, nrows=0)
    day_cols = _day_columns(list(header.columns))
    if not day_cols:
        raise ValueError(f"No d_* day columns found in {path}")
    dtypes = {c: np.int32 for c in day_cols}
    sales = pd.read_csv(path, dtype=dtypes)
    meta_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    matrix = sales[day_cols].to_numpy(dtype=np.int32)
    return sales[meta_cols].reset_index(drop=True), day_cols, matrix


def _daily_price_lookup(
    sales_meta: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    day_cols: list[str],
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """Build a (n_series, n_days) daily price matrix + an (item, store) row map.

    Weekly ``sell_price`` is expanded to daily via the calendar's d_* -> wm_yr_wk
    mapping. Days with no known price are NaN.
    """
    day_week = calendar.set_index("d").loc[day_cols, "wm_yr_wk"].to_numpy()
    pivot = prices.pivot_table(
        index=["item_id", "store_id"],
        columns="wm_yr_wk",
        values="sell_price",
        aggfunc="first",
    )
    week_to_col = {int(wk): i for i, wk in enumerate(pivot.columns.to_numpy())}
    # Column index into the pivot for each day; -1 where the week is absent.
    day_to_col = np.array([week_to_col.get(int(wk), -1) for wk in day_week], dtype=np.int64)
    pivot_values = pivot.to_numpy(dtype=np.float64)  # (n_keys, n_weeks), NaN where missing
    key_to_row = {
        (str(item), str(store)): row for row, (item, store) in enumerate(pivot.index.to_numpy())
    }
    n_series = len(sales_meta)
    daily_price = np.full((n_series, len(day_cols)), np.nan, dtype=np.float64)
    has_col = day_to_col >= 0
    cols = day_to_col[has_col]
    for i in range(n_series):
        key = (str(sales_meta.iat[i, 0]), str(sales_meta.iat[i, 3]))  # item_id, store_id
        row = key_to_row.get(key)
        if row is None:
            continue
        weekly = pivot_values[row]
        daily_price[i, has_col] = weekly[cols]
    return daily_price, key_to_row


def _compute_metrics(
    sales_meta: pd.DataFrame,
    matrix: np.ndarray,
    daily_price: np.ndarray,
) -> pd.DataFrame:
    """Per-series ADI / CV2 / SB quadrant / seasonal-strength / promo-intensity."""
    n_series = matrix.shape[0]
    records: list[dict[str, object]] = []
    for i in range(n_series):
        active = classify.trim_leading_zeros(matrix[i].astype(np.float64))
        n_active = int(active.shape[0])
        nonzero = int(np.count_nonzero(active))
        adi = classify.adi(active)
        cv2 = classify.cv_squared(active)
        # Promo uses the same trimmed window so price/demand stay aligned.
        price_full = daily_price[i]
        price_active = price_full[price_full.shape[0] - n_active :] if n_active > 0 else price_full
        records.append(
            {
                "item_id": str(sales_meta.iat[i, 0]),
                "dept_id": str(sales_meta.iat[i, 1]),
                "cat_id": str(sales_meta.iat[i, 2]),
                "store_id": str(sales_meta.iat[i, 3]),
                "state_id": str(sales_meta.iat[i, 4]),
                "n_active": n_active,
                "n_nonzero": nonzero,
                "mean": float(np.mean(active)) if n_active > 0 else 0.0,
                "std": float(np.std(active)) if n_active > 0 else 0.0,
                "zero_fraction": (1.0 - nonzero / n_active) if n_active > 0 else 1.0,
                "adi": adi,
                "cv2": cv2,
                "sb_quadrant": classify.classify_metrics(adi, cv2),
                "seasonal_strength": _seasonal_strength(active),
                "promo_intensity": _promo_intensity(active, price_active),
            }
        )
    return pd.DataFrame.from_records(records)


def _select_archetypes(metrics: pd.DataFrame) -> dict[str, int]:
    """Pick one distinct row index per archetype (deterministic, tie-broken)."""
    eligible = metrics[(metrics["n_nonzero"] >= MIN_NONZERO) & (metrics["n_active"] >= MIN_ACTIVE)]
    if eligible.empty:
        raise ValueError("No eligible series after applying MIN_NONZERO / MIN_ACTIVE floors")
    chosen: dict[str, int] = {}
    used: set[int] = set()

    def _pick(pool: pd.DataFrame, by: str, ascending: bool) -> int:
        ranked = pool[~pool.index.isin(used)].sort_values(
            [by, "item_id", "store_id"], ascending=[ascending, True, True]
        )
        if ranked.empty:
            raise ValueError(f"No remaining eligible series to pick for sort key {by!r}")
        idx = int(ranked.index[0])
        used.add(idx)
        return idx

    smooth_pool = eligible[eligible["sb_quadrant"] == "smooth"]
    chosen["smooth"] = _pick(smooth_pool if not smooth_pool.empty else eligible, "mean", False)

    intermittent_pool = eligible[eligible["sb_quadrant"] == "intermittent"]
    chosen["intermittent"] = _pick(
        intermittent_pool if not intermittent_pool.empty else eligible, "mean", True
    )

    # Seasonal: a genuine recurring pattern needs a regularly-selling series, so
    # restrict to NON-intermittent quadrants (intermittent "seasonality" is
    # mostly zero-gap noise). Pick the strongest weekly/yearly autocorrelation.
    seasonal_pool = eligible[eligible["sb_quadrant"].isin(["smooth", "erratic"])]
    chosen["seasonal"] = _pick(
        seasonal_pool if not seasonal_pool.empty else eligible, "seasonal_strength", False
    )

    # Promotional: price-driven demand only makes sense for an item that sells
    # most days, so restrict to a low zero-fraction. Pick the strongest
    # discount-frequency x demand-lift signal.
    promo_pool = eligible[eligible["zero_fraction"] < 0.5]
    if not promo_pool.empty and float(promo_pool["promo_intensity"].max()) > 0.0:
        chosen["promotional"] = _pick(promo_pool, "promo_intensity", False)
    else:
        print("WARNING: no price-based promo signal; falling back to highest CV2")
        fallback = promo_pool if not promo_pool.empty else eligible
        chosen["promotional"] = _pick(fallback, "cv2", False)
    return chosen


def _write_outputs(
    metrics: pd.DataFrame,
    matrix: np.ndarray,
    chosen: dict[str, int],
) -> None:
    """Write the 4 archetype parquets + the provenance manifest."""
    manifest: dict[str, dict[str, object]] = {}
    for archetype in ARCHETYPES:
        idx = chosen[archetype]
        active = classify.trim_leading_zeros(matrix[idx].astype(np.float64))
        out_path = OUT_DIR / f"archetype_{archetype}.parquet"
        pd.DataFrame({"demand": active}).to_parquet(out_path, index=False)
        row = metrics.loc[idx]
        manifest[archetype] = {
            "parquet": out_path.name,
            "item_id": str(row["item_id"]),
            "dept_id": str(row["dept_id"]),
            "cat_id": str(row["cat_id"]),
            "store_id": str(row["store_id"]),
            "state_id": str(row["state_id"]),
            "sb_quadrant": str(row["sb_quadrant"]),
            "n_rows": int(active.shape[0]),
            "mean": round(float(row["mean"]), 6),
            "std": round(float(row["std"]), 6),
            "zero_fraction": round(float(row["zero_fraction"]), 6),
            "adi": round(float(row["adi"]), 6),
            "cv2": round(float(row["cv2"]), 6),
            "seasonal_strength": round(float(row["seasonal_strength"]), 6),
            "promo_intensity": round(float(row["promo_intensity"]), 6),
        }
        print(
            f"  {archetype:13s} <- {row['item_id']} @ {row['store_id']} "
            f"[SB={row['sb_quadrant']}] n={active.shape[0]} mean={float(row['mean']):.3f} "
            f"zero_frac={float(row['zero_fraction']):.3f}"
        )
    manifest_path = OUT_DIR / "archetypes_manifest.json"
    manifest_doc = {
        "description": (
            "Curated real Walmart M5 SKUs, one per demand archetype, "
            "bundled for the EmpiricalDemand arm. Demand is replayed IID "
            "(rng.choice) — marginal-preserving, not temporal. Leading "
            "(pre-introduction) zeros are trimmed. SB cutoffs: ADI=1.32, CV2=0.49."
        ),
        "source": "M5 Forecasting - Accuracy (Makridakis/Walmart) sales_train_validation.csv",
        "archetypes": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n")
    print(f"  manifest      -> {manifest_path.relative_to(REPO_ROOT)}")


def main() -> None:
    sales_path = RAW_DIR / "sales_train_validation.csv"
    calendar_path = RAW_DIR / "calendar.csv"
    prices_path = RAW_DIR / "sell_prices.csv"
    for path in (sales_path, calendar_path, prices_path):
        if not path.exists():
            raise SystemExit(
                f"Missing required input: {path}\n"
                f"Place the raw M5 CSVs in {RAW_DIR} (gitignored) and re-run."
            )

    print(f"Loading sales from {sales_path.relative_to(REPO_ROOT)} ...")
    sales_meta, day_cols, matrix = _load_sales(sales_path)
    print(f"  {matrix.shape[0]} series x {len(day_cols)} days")

    print("Aligning weekly prices to daily demand ...")
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    daily_price, _ = _daily_price_lookup(sales_meta, calendar, prices, day_cols)

    print("Computing per-series ADI / CV2 / seasonal-strength / promo-intensity ...")
    metrics = _compute_metrics(sales_meta, matrix, daily_price)

    print("Selecting one representative SKU per archetype:")
    chosen = _select_archetypes(metrics)

    _write_outputs(metrics, matrix, chosen)
    print("Done.")


if __name__ == "__main__":
    main()
