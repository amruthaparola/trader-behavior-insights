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


def cmd_risk(_args: argparse.Namespace) -> None:
    """Step 4: risk-adjusted performance by regime and by trader."""
    import pandas as pd

    from . import data
    from . import metrics as mt

    ds = data.load_dataset()
    opts = ("display.width", 160, "display.max_columns", 30)

    reg = mt.regime_risk_table(ds.account_day, ds.closes, ds.sentiment)
    daily_cols = ["n_days", "total_pnl", "sharpe", "sharpe_se", "psr_vs_0", "sortino",
                  "max_drawdown", "var95", "cvar95", "skew", "excess_kurtosis"]
    trade_cols = ["n_trades", "hit_rate", "avg_win", "avg_loss", "payoff_ratio",
                  "breakeven_hit_rate", "profit_factor", "expectancy"]
    with pd.option_context(*opts, "display.float_format", "{:,.2f}".format):
        print("1) Book-level DAILY risk by same-day regime (USD; Sharpe annualised with sqrt(365))")
        print(reg[daily_cols].to_string())
        print("\n2) TRADE-level stats by regime (closing trades, net of closing fee, USD)")
        print(reg[trade_cols].to_string())
        print("\n3) Fee impact on the whole book")
        print(mt.fee_impact(ds.account_day).to_string())
        print("\n4) Trader leaderboard (sorted by Sharpe)")
        print(mt.trader_leaderboard(ds.account_day, ds.closes).to_string())


def cmd_models(_args: argparse.Namespace) -> None:
    """Step 5: predictive regressions with lagged sentiment."""
    import pandas as pd

    from . import data
    from . import models as md

    ds = data.load_dataset()
    m = ds.market_day
    opts = ("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format)

    with pd.option_context(*opts):
        for winsor in (True, False):
            res = md.time_series_models(m, winsor=winsor)
            tag = "winsorised 1/99%" if winsor else "raw"
            print(f"1) Daily return (bps) on yesterday's sentiment / 10  [{tag}, HAC lags={res.attrs['hac_lags']}]")
            print(res.to_string(), "\n")

        reg = md.regime_dummy_model(m)
        print("2) Yesterday's regime as dummies (baseline Neutral), HAC SEs")
        print(reg.to_string())
        print(f"   joint Wald test, all regime effects = 0:  p = {reg.attrs['joint_p']:.3f}\n")

        print("3) Panel: account-day return on yesterday's sentiment / 10")
        print(md.panel_models(ds.account_day).to_string(), "\n")

        print("4a) Notebook-style transitions (yesterday -> today), with day counts")
        tt = md.transition_table(m)
        print(tt[tt["n_days"] > 0].to_string(), "\n")
        print("4b) Predictive: did the regime worsen/improve from t-2 to t-1?  (baseline unchanged, HAC)")
        print(md.change_direction_model(m).to_string(), "\n")

    print("5) Out-of-sample: expanding-window forecast vs historical mean")
    for k, v in md.out_of_sample_r2(m).items():
        print(f"   {k:<22} {v:.4f}" if isinstance(v, float) else f"   {k:<22} {v}")


def cmd_prices(_args: argparse.Namespace) -> None:
    """Download daily BTC candles once (needed by the backtest)."""
    from . import prices

    df = prices.download_btc()
    print(f"saved {len(df)} days ({df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}) "
          f"from {df['source'].iloc[0]} to {prices.BTC_CSV}")


def cmd_backtest(_args: argparse.Namespace) -> None:
    """Step 6: sentiment strategies on BTC with costs and an out-of-sample split."""
    import pandas as pd

    from . import backtest as bt

    d = bt.load_frame()
    opts = ("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format)
    print(f"BTC daily {d['date'].min():%Y-%m-%d} -> {d['date'].max():%Y-%m-%d}; "
          f"out-of-sample from {bt.TEST_START:%Y-%m-%d}; costs {bt.COST_BPS:.0f} bps per unit turnover\n")

    with pd.option_context(*opts):
        print("1) Rule performance (Sharpe annualised sqrt(365); max_dd and cagr as fractions)")
        print(bt.evaluate_rules(d).to_string(), "\n")

        print("2) Out-of-sample Sharpe vs transaction cost")
        print(bt.cost_sensitivity(d).to_string(), "\n")

        ts = bt.threshold_search(d)
        print("3) Tuning 'long when score_{t-1} < k' on in-sample data only")
        print(ts.to_string())
        k = ts.attrs["best_k_in_sample"]
        print(f"   best k in-sample = {k}  ->  OOS Sharpe {ts.loc[k, 'oos_sharpe']:.3f}")
        print(f"   Spearman rank corr(IS Sharpe, OOS Sharpe) across k = {ts.attrs['rank_corr_is_oos']:.3f}\n")

    print("4) Timing test: rule vs the same positions rotated in time (circular shift)")
    oos = d["date"] >= bt.TEST_START
    for name, rule in bt.RULES.items():
        if name == "Buy & hold":
            continue
        pos = rule(d)
        full = bt.timing_test(pos, d["ret"])
        test = bt.timing_test(pos[oos.to_numpy()], d.loc[oos, "ret"])
        print(f"   {name:<26} full: SR {full['sharpe']:.2f} vs null {full['null_mean']:.2f}, p={full['p']:.3f}"
              f"   | OOS: SR {test['sharpe']:.2f} vs null {test['null_mean']:.2f}, p={test['p']:.3f}")

    path = bt.plot_equity(d)
    print(f"\nFigure saved: {path}")


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

    p_risk = sub.add_parser("risk", help="Step 4: risk metrics by regime and by trader")
    p_risk.set_defaults(func=cmd_risk)

    p_models = sub.add_parser("models", help="Step 5: predictive regressions on lagged sentiment")
    p_models.set_defaults(func=cmd_models)

    p_prices = sub.add_parser("prices", help="download daily BTC candles (once)")
    p_prices.set_defaults(func=cmd_prices)

    p_bt = sub.add_parser("backtest", help="Step 6: sentiment-signal backtest on BTC")
    p_bt.set_defaults(func=cmd_backtest)

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
