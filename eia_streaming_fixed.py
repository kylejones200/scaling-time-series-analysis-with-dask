import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

import logging
import yaml

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(config.get('data', {}).get('seed', 42))

plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()


def load_eia_series(csv_path: str, freq: str = "MS") -> pd.Series:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    # Assume ISO date (YYYY-MM-01) and comma-separated first two columns
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=['date', 'value'], sep=',')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    out = df.dropna()
    return out.sort_values('date').set_index('date')['value'].asfreq(freq)


def main():
    csv_path = "Net_generation_United_States_all_sectors_monthly.csv"
    s = load_eia_series(csv_path, freq="MS").astype(float)

    history = deque(maxlen=60)
    times, ytrue, yhat = [], [], []

    for t, val in s.items():
        if len(history) >= 24:
            y_hist = pd.Series(list(history))
            model = sm.tsa.statespace.SARIMAX(
                y_hist, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = model.fit(disp=False)
            pred = float(res.forecast(steps=1).iloc[0])
            times.append(t); yhat.append(pred); ytrue.append(val)
        history.append(val)

    ytrue_s = pd.Series(ytrue, index=pd.to_datetime(times))
    yhat_s = pd.Series(yhat, index=pd.to_datetime(times))
    logger.error("Online one-step MAE:", mean_absolute_error(ytrue_s.values, yhat_s.values))

    # Tufte-style focus: history 2024, vertical line at 2025-01-01, Jan–Aug 2025 actuals vs forecast
    start_2024 = pd.Period('2024-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period('2024-12', freq='M').start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period('2025-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period('2025-08', freq='M').start_time + pd.offsets.MonthBegin(0)

    y_hist = s.loc[start_2024:end_2024]
    y_act = s.loc[jan_2025:aug_2025]
    f = yhat_s.loc[jan_2025:aug_2025]

    # Residual-based band on the same window
    common = ytrue_s.loc[jan_2025:aug_2025].align(f, join='inner')
    if all(len(x) for x in common):
        resid = common[0] - common[1]
        sigma = float(resid.std(ddof=1)) if len(resid) else 0.0
    else:
        sigma = 0.0
    upper = f + 1.96 * sigma
    lower = f - 1.96 * sigma

    fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [10, 5])))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    if len(f):
        ax.fill_between(f.index, lower.values, upper.values, color="#000000", alpha=0.06, linewidth=0)
        ax.plot(f.index, f.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    # Staggered end labels to reduce overlap
    y_min, y_max = (min(s.min(), f.min()) if len(f) else s.min()), (max(s.max(), f.max()) if len(f) else s.max())
    yrng = max(1.0, y_max - y_min)
    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6, -0.02*yrng), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666', clip_on=False)
    if len(y_act):
        ax.annotate('Actual (Jan–Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6, 0.02*yrng), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444', clip_on=False)
    if len(f):
        ax.annotate('Forecast', xy=(f.index[-1], f.values[-1]), xytext=(6, 0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000', clip_on=False)

    ax.set_title('EIA Net Generation — Online SARIMAX one-step forecast Jan–Aug 2025')
    ax.set_xlabel('')
    ax.set_ylabel('')
    save_fig("eia_streaming_last.png")

if __name__ == "__main__":
    main()
