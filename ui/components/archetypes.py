"""M5 demand-archetype manifest loader.

Reads ``data/m5/archetypes_manifest.json`` — four curated real Walmart M5 SKUs, one per
demand archetype (smooth · intermittent · seasonal · promotional) — into typed :class:`Archetype`
records. Streamlit-free pure logic: the bridge from the M2 archetype data to the
config-builder empirical demand arm and, later, the Policy Advisor.

``history_path`` is **repo-relative** (``data/m5/…``), never absolute: it is stored in the
``RunConfig`` and folded into the reproducibility hash, so an absolute path would make hashes
machine-specific and non-portable. The parquet's existence / schema / content is NOT checked here —
that happens later, loudly, at ``EmpiricalDemand.__init__`` time (Pydantic and this loader stay
I/O-light); ``load_archetypes`` only reads the small manifest.

Note the ``name`` vs ``sb_quadrant`` divergence: ``name`` is the curated *business* archetype, while
``sb_quadrant`` is the data-driven *statistical* Syntetos-Boylan class — they need not match (the
"seasonal" SKU is SB "erratic", the "promotional" SKU is SB "lumpy"). That is a teaching point, not
a bug, and the UI surfaces both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# One constant, two roles: joined to the repo root it locates the bundled files (cwd-robust read);
# as a literal prefix it yields the cwd-relative history_path stored in configs (portable hash).
_M5_DIRNAME: Final = "data/m5"
_MANIFEST: Final = Path(__file__).resolve().parents[2] / _M5_DIRNAME / "archetypes_manifest.json"


@dataclass(frozen=True)
class Archetype:
    """One curated M5 demand archetype: a replayable history path + its demand-shape stats.

    - ``name``: the curated *business* archetype (smooth / intermittent / seasonal / promotional).
    - ``history_path``: repo-relative parquet path for ``EmpiricalDemandConfig(history_path=…)``.
    - ``sb_quadrant``: the data-driven *statistical* Syntetos-Boylan class — **may differ from**
      ``name`` (e.g. the seasonal SKU is "erratic", the promotional SKU is "lumpy").
    - the remaining fields are the SKU's demand-shape statistics (the CV²/ADI quadrant plus
      mean / std / zero-fraction / seasonal-strength / promo-intensity over its trimmed history).
    """

    name: str
    history_path: str
    item_id: str
    sb_quadrant: str
    n_rows: int
    mean: float
    std: float
    cv2: float
    adi: float
    zero_fraction: float
    seasonal_strength: float
    promo_intensity: float


def load_archetypes() -> dict[str, Archetype]:
    """Load the four M5 archetypes from the bundled manifest, keyed by name in manifest order.

    Pure read of ``data/m5/archetypes_manifest.json`` (no parquet I/O); returns a fresh dict each
    call. Raises ``FileNotFoundError`` if the bundled manifest is missing (fail-loud).
    """
    raw = json.loads(_MANIFEST.read_text())["archetypes"]
    return {
        name: Archetype(
            name=name,
            history_path=f"{_M5_DIRNAME}/{e['parquet']}",
            item_id=str(e["item_id"]),
            sb_quadrant=str(e["sb_quadrant"]),
            n_rows=int(e["n_rows"]),
            mean=float(e["mean"]),
            std=float(e["std"]),
            cv2=float(e["cv2"]),
            adi=float(e["adi"]),
            zero_fraction=float(e["zero_fraction"]),
            seasonal_strength=float(e["seasonal_strength"]),
            promo_intensity=float(e["promo_intensity"]),
        )
        for name, e in raw.items()
    }
