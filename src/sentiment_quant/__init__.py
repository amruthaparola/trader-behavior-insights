"""sentiment_quant: does Bitcoin market sentiment explain or predict trader performance?

Modules (added step by step, see the learning plan):
    config       - paths and constants shared by every module
    data         - load, clean and join trades with the Fear & Greed index   (Step 2)
    stats_tests  - dependence-aware hypothesis tests                         (Step 3)
    metrics      - risk / performance metrics (Sharpe, drawdown, CVaR, ...)  (Step 4)
    models       - predictive regressions with HAC / clustered SEs           (Step 5)
    backtest     - sentiment-signal strategy backtest                        (Step 6)
"""

__version__ = "0.1.0"
