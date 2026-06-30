"""Command-line entry point: run a simulation (or Monte Carlo) from a YAML ``RunConfig``.

Usage::

    python -m inventory_twin.cli run        --config CONFIG [--out OUT] [--quiet]
    python -m inventory_twin.cli montecarlo --config CONFIG [--reps N] [--jobs J] [--out OUT]

``run`` wires the config through the engine + policy and writes the Ledger CSV;
``montecarlo`` runs N seeded replications and writes the per-replication KPI
distribution CSV. Re-running either with the
same config yields a byte-identical CSV — the reproducibility gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analytics.monte_carlo import monte_carlo
from core.config import RunConfig
from experiments import single_run


def _run(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(Path(args.config).read_text())
    df = single_run.run(config)
    out_path = Path(args.out) if args.out else Path(f"{config.config_hash()[:12]}.csv")
    df.to_csv(out_path, index=False)

    if not args.quiet:
        total_cost = (
            df["holding_cost"] + df["ordering_cost"] + df["purchase_cost"] + df["stockout_cost"]
        ).sum()
        print(f"Run:        {config.name or 'unnamed'}")
        print(f"Hash:       {config.config_hash()[:12]}")
        print(f"Horizon:    {config.simulation.horizon} periods")
        print(f"Total cost: ${total_cost:,.2f}")
    print(f"Wrote ledger to: {out_path}")
    return 0


def _montecarlo(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(Path(args.config).read_text())
    df = monte_carlo(config, args.reps, n_jobs=args.jobs)
    out_path = Path(args.out) if args.out else Path(f"{config.config_hash()[:12]}_mc.csv")
    df.to_csv(out_path, index=False)

    if not args.quiet:
        cost = df["total_cost"]
        fill = df["fill_rate"]
        print(f"Run:          {config.name or 'unnamed'}")
        print(f"Hash:         {config.config_hash()[:12]}")
        print(f"Replications: {args.reps}")
        print(
            f"Total cost:   mean ${cost.mean():,.2f}  "
            f"p5 ${cost.quantile(0.05):,.2f}  p95 ${cost.quantile(0.95):,.2f}"
        )
        print(f"Fill rate:    mean {fill.mean():.3f}  p5 {fill.quantile(0.05):.3f}")
    print(f"Wrote KPI distribution to: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inventory_twin.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single simulation from a YAML config")
    p_run.add_argument("--config", required=True, help="Path to RunConfig YAML")
    p_run.add_argument("--out", help="Output CSV path (default: <hash12>.csv in CWD)")
    p_run.add_argument("--quiet", action="store_true", help="Suppress the human summary")
    p_run.set_defaults(func=_run)

    p_mc = sub.add_parser(
        "montecarlo", help="Run N seeded replications; write the per-rep KPI distribution"
    )
    p_mc.add_argument("--config", required=True, help="Path to RunConfig YAML")
    p_mc.add_argument(
        "--reps", type=int, default=1000, help="Number of replications (default: 1000)"
    )
    p_mc.add_argument(
        "--jobs", type=int, default=-1, help="Parallel joblib workers (default: -1 = all cores)"
    )
    p_mc.add_argument("--out", help="Output CSV path (default: <hash12>_mc.csv in CWD)")
    p_mc.add_argument("--quiet", action="store_true", help="Suppress the human summary")
    p_mc.set_defaults(func=_montecarlo)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
