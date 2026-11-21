"""Command-line entry point:  python -m sentiment_quant <command>"""

import argparse

from . import __version__, config


def cmd_info(_args: argparse.Namespace) -> None:
    """Print where the project expects its data and writes its outputs."""
    print(f"sentiment_quant {__version__}")
    for name in ("PROJECT_ROOT", "TRADES_CSV", "SENTIMENT_CSV", "FIGURES_DIR", "RESULTS_JSON"):
        path = getattr(config, name)
        status = "found" if path.exists() else "missing"
        print(f"  {name:<14} {path}  [{status}]")


def cmd_data(_args: argparse.Namespace) -> None:
    """Build the dataset and print a sanity summary + the win-rate correction."""
    import pandas as pd

    from . import data

    ds = data.load_dataset()
    t = ds.trades
    print("Dataset summary")
    print(f"  fills            {len(t):,}")
    print(f"  closing fills    {len(ds.closes):,}  ({len(ds.closes) / len(t):.1%} of fills)")
    print(f"  accounts         {t['account'].nunique()}")
    print(f"  UTC days         {t['date'].nunique()}  ({t['date'].min():%Y-%m-%d} -> {t['date'].max():%Y-%m-%d})")
    print(f"  account-days     {len(ds.account_day):,}")
    print(f"  gross PnL        ${t['closed_pnl'].sum():,.0f}")
    print(f"  fees             ${t['fee'].sum():,.0f}")
    print(f"  net PnL          ${t['net_pnl'].sum():,.0f}")

    print("\nWin rate by same-day sentiment: notebook definition vs closing trades only")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(data.win_rate_comparison(t).to_string())


def cmd_stats(_args: argparse.Namespace) -> None:
    """Step 3: naive vs dependence-aware tests of 'performance differs by regime'."""
    import pandas as pd

    from . import data
    from . import stats_tests as st

    ds = data.load_dataset()
    fmt = {"display.width": 140, "display.max_columns": 20}

    print("1) What the notebook did: every closing trade treated as independent")
    for k, v in st.naive_trade_level_tests(ds.closes).items():
        print(f"   {k:<28} {v:.4g}" if isinstance(v, float) else f"   {k:<28} {v}")

    print("\n2) How much do i.i.d. standard errors understate uncertainty? (win rate)")
    with pd.option_context(*sum(fmt.items(), ()), "display.float_format", "{:.4f}".format):
        print(st.design_effect_table(ds.closes).to_string())

    print("\n3) Regime estimates with 95% CIs from a day-cluster bootstrap")
    with pd.option_context(*sum(fmt.items(), ()), "display.float_format", "{:.3f}".format):
        print(st.regime_bootstrap_table(ds.closes).to_string())

    print("\n4) Global test on daily net PnL per $ traded (one observation per day)")
    for label in ("sentiment", "sentiment_lag1"):
        res = st.global_tests(ds.market_day, label_col=label)
        print(f"   [{label}]")
        for k, v in res.items():
            print(f"     {k:<24} {v:.4g}" if isinstance(v, float) else f"     {k:<24} {v}")

    print("\n5) Pairwise regime comparisons on daily data, Holm-corrected")
    with pd.option_context(*sum(fmt.items(), ()), "display.float_format", "{:.3f}".format):
        print(st.pairwise_regime_tests(ds.market_day).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentiment_quant",
        description="Bitcoin sentiment vs trader performance: reproducible quant analysis.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_info = sub.add_parser("info", help="show configured data/output paths")
    p_info.set_defaults(func=cmd_info)

    p_data = sub.add_parser("data", help="build the cleaned dataset and print a summary")
    p_data.set_defaults(func=cmd_data)

    p_stats = sub.add_parser("stats", help="Step 3: dependence-aware hypothesis tests")
    p_stats.set_defaults(func=cmd_stats)

    # Later steps add: `run` (full pipeline, Step 7).
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
