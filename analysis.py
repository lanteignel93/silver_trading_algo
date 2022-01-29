import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import os
from data import *
import macdsignal as ms
from backtester import BackTest as BT
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.model_selection import train_test_split
from scipy.stats import norm, normaltest, skewtest, kurtosistest
import scipy.stats
import seaborn as sns
import datetime
from statsmodels.tsa.stattools import adfuller
import warnings
from scipy.optimize import minimize





ROOT_DIR = os.getcwd()
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
DATA_PATH = os.path.join(ROOT_DIR, "data")
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def macd(df,m1, m2,window_type):
    """
    df: Dataframe compute MACD
    M1: Short-Window,
    M2: Long-Window,
    window_type: (ema/sma) Exponential Moving Average / Simple Moving Average
    """
    if window_type == 'ema':
        return df.ewm(span=m1).mean() - df.ewm(span=m2).mean()
    elif window_type == 'sma':
        return df.rolling(m1).mean() - df.rolling(m2).mean()


def plot_macd_silver(indicator, y, small_window, large_window, window_type = 'ema'):
    macd_vals = macd(indicator, small_window, large_window ,window_type)
    train_df = pd.DataFrame({"Silver": y, 'Indicator MACD': macd_vals})

    fig, ax1 = plt.subplots(figsize = (16,6))
    fig.suptitle('Silver Price and ({},{}) MACD'.format(small_window, large_window),fontsize = 20)
    color = 'tab:red'
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Silver', color=color, fontsize=14)
    ax1.plot(train_df.loc[:,'Silver'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Indicator MACD', color=color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(train_df.loc[:,'Indicator MACD'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    save_fig('silver_macd')
    plt.plot()


def pred_indicator(x,y,M):
    '''
    x = factors
    y = silver price
    M = window size. Note data is hourly
    '''
    x_ret = x.pct_change()
    x_ret['const'] = 1
    y_ret = y.pct_change()
    rols = RollingOLS(y_ret, x_ret, M)
    rres = rols.fit()
    params = rres.params.copy().shift(1).dropna()
    x_ret = x_ret[x_ret.index.isin(params.index)]
    pred_y_ret = (params * x_ret).sum(axis=1)
    pred_y = (1+pred_y_ret) * y
    return pred_y

def plot_silver_volume(df):
    fig, ax = plt.subplots(1,figsize = (14,8))
    ax.set_title("Silver Trading Volume over Time", fontsize = 18)
    ax.set_ylabel("Volume", fontsize = 16)
    ax.set_xlabel("Date",  fontsize = 16)
    ## JS Change. Original code threw error on the X arg being a timestamp
    x = df['SI1:Volume'].index.values
    # x = silver['SI1:Volume'].index
    y = df['SI1:Volume']
    ax.plot(x,y, label = 'Volume')
    ax.plot(x,[y.mean()]*len(x), ls = '--', color = 'red', label = 'Mean')
    ax.legend(fontsize = 16)
    print("Mean Volume: {}".format(int(y.mean())))
    save_fig('silver_volume')
    plt.show()

def print_vol_distr_stats(df):
    y = df['SI1:Volume']
    pct_under = len(y[y<y.mean()])/len(y)
    pct_under_exp = len(y[y<1000])/len(y)
    print("Percentage of the time trading volume is below the mean: {:.2f}%".format(100*pct_under))
    print("Percentage of the time trading volume below our expected trading size: {:.2f}%".format(100*pct_under_exp))


def compute_window_max_min(df, vol_adjust, price_type = 'close'):
    if price_type == 'h/l':
        h = 'high'
        l = 'low'
    else:
        h = 'close'
        l = 'close'
    tmp = df.copy()
    tmp['SI1:Volume'].fillna(method='ffill', inplace = True)
    tmp['VolumeCumSum'] = tmp['SI1:Volume'].cumsum()
    tmp['VolumeInd'] = np.where((tmp['VolumeCumSum'] % vol_adjust).diff() < 0, 1,0)
    tmp['WindowMax'] = np.nan
    tmp['WindowMin'] = np.nan

    curr_max = -np.float('inf')
    curr_min = np.float('inf')

    for t in tmp.index:
        if tmp.loc[t,'VolumeInd'] == 0:
            if tmp.loc[t,'SI1:{}'.format(h)] > curr_max:
                curr_max = tmp.loc[t,'SI1:{}'.format(h)]

            if tmp.loc[t,'SI1:{}'.format(l)] < curr_min:
                curr_min = tmp.loc[t,'SI1:{}'.format(l)]

        elif tmp.loc[t,'VolumeInd'] == 1:
            if tmp.loc[t,'SI1:{}'.format(h)] > curr_max:
                curr_max = tmp.loc[t,'SI1:{}'.format(h)]

            if tmp.loc[t,'SI1:{}'.format(l)] < curr_min:
                curr_min = tmp.loc[t,'SI1:{}'.format(l)]

            tmp.loc[t, 'WindowMax'] = curr_max
            tmp.loc[t, 'WindowMin'] = curr_min

            curr_max = -np.float('inf')
            curr_min = np.float('inf')

    tmp['WindowMax'].fillna(method='bfill', inplace = True)
    tmp['WindowMin'].fillna(method='bfill', inplace = True)
    return tmp

def average_slippage_cost(df, volume):
    tmp_df = compute_window_max_min(df, volume, price_type = 'h/l')
    tmp_df['Max/Close'] = 100*abs(tmp_df['WindowMax']/tmp_df['SI1:close'] - 1)
    tmp_df['Min/Close'] = 100*abs(tmp_df['WindowMin']/tmp_df['SI1:close'] - 1)
    max_slip_hl, min_slip_hl = tmp_df['Max/Close'].mean(), tmp_df['Min/Close'].mean()

    tmp_df = compute_window_max_min(df, volume, price_type = 'close')
    tmp_df['Max/Close'] = 100*abs(tmp_df['WindowMax']/tmp_df['SI1:close'] - 1)
    tmp_df['Min/Close'] = 100*abs(tmp_df['WindowMin']/tmp_df['SI1:close'] - 1)
    max_slip_close, min_slip_close = tmp_df['Max/Close'].mean(), tmp_df['Min/Close'].mean()

    print("Average Slippage for Buying Price: {:.2f}% and Selling Price: {:.2f}% using High/Low of next {} traded contracts".format(max_slip_hl, min_slip_hl, volume))
    print("Average Slippage for Buying Price: {:.2f}% and Selling Price: {:.2f}% using Close of next {} traded contracts".format(max_slip_close, min_slip_close, volume))


def distr_plots(r):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    r.plot(ax=axs[0], title='Plot of Returns for Strategy', grid=True)
    r.plot(kind='hist', bins=50, log = True, ax=axs[1], title='Distribution of Returns for Strategy', grid=True)
    axs[1].axvline(r.median(), color='red', linestyle='--')
    save_fig('strat_dist_plot')
    plt.show();
    return

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol



def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index,
       the previous peaks, and
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index,
                         "Previous Peak": previous_peaks,
                         "Drawdown": drawdowns})


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r.dropna(), level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gausian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def summary_stats(r, riskfree_rate=0.01):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    spx = pd.read_csv(DATA_PATH +'\\' + "SPX_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    spx = spx.rename(columns = rename_cols(spx,"SPX"))
    bench_ret = pd.read_pickle(DATA_PATH +'//bench_ret.pkl')
    r['SPX'] = spx.between_time('17:00', '17:30').set_index(spx.between_time("17:00", '17:30').index.date)['SPX:close'].pct_change().replace([np.inf, np.nan],0)
    r['GSCI'] = bench_ret['GSCI']
    r['SI1'] = bench_ret['SI1']
    ann_r = r.aggregate(annualize_rets, periods_per_year=252)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=252)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=252)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    distr_plots(r['TS'])
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd,
        "Market Beta": r.corr()['SPX'],
        'GSCI Beta' : r.corr()['GSCI'],
        'Silver Beta' : r.corr()['SI1'],
        'Treynor Ratio' : ann_r/r.corr()['SPX']
    })

def ff_plot(ff, ret):
    # Plot Returns
    df = ff.copy()
    df = df[df.index.isin(ret.index)]
    sns.set_style("whitegrid")
    df['TS'] = ret
    df['TS'].fillna(0)
    (df+1).cumprod().plot(figsize = (16,6))
#     (ret.cumsum()+1).plot(label = 'TS')
    plt.title("Returns vs Fama French Factors", fontsize = 18)
    plt.ylabel("Cumulative Returns")
    save_fig('ff_plot')
    plt.show();
    return

def ff_corr(ff, ret):
    df = ff.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    df['TS'].fillna(0)
    return df[['HML','SMB','Mkt-RF','TS']].corr()

def ff_regr_plots(ff, ret):
    # Plot
    df = ff.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    sns.regplot(x=df['TS'], y=df['Mkt-RF'], color='blue', marker='+', ax = axs[0])
    axs[0].title.set_text('TS vs Market Returns')
    axs[0].set_xlabel('Daily TS Returns')
    axs[0].set_ylabel('Market Returns')

    sns.regplot(x=df['TS'], y=df['SMB'], color='magenta', marker='+', ax = axs[1])
    axs[1].title.set_text('TS vs SMB Returns')
    axs[1].set_xlabel('Daily TS Returns')
    axs[1].set_ylabel('SMB Returns')

    sns.regplot(x=df['TS'], y=df['HML'], color='green', marker='+', ax = axs[2])
    axs[2].title.set_text('TS vs HML Returns')
    axs[2].set_xlabel('Daily TS Returns')
    axs[2].set_ylabel('HML Returns')


    plt.tight_layout()
    save_fig('ff_regr_plots')
    plt.show();
    return

def ff_OLS(ff, ret):
    df = ff.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    factors = df[['Mkt-RF', 'SMB', 'HML']]
    rhs = sm.add_constant(factors)
    lhs = df['TS']
    res = sm.OLS(lhs, rhs, missing='drop').fit()
    display(res.summary())
    return

def ff_analysis(ff, ret):
    ff_plot(ff, ret)
    ff_regr_plots(ff, ret)
    ff_OLS(ff, ret)
    res = ff_corr(ff, ret)
    display(res)

    return

def optimize_TS(x, df, y_train,indicator_val_rolling, slip_train, lot_size):
    macd_m1 = x[0]
    macd_m2 = x[1]
    macd_enter = np.repeat(x[2], len(df))
    macd_exit = np.repeat(x[3], len(df))
    rsi_upper = x[4]
    rsi_lower = x[5]
    start = '2020-01-01'
    start_trading = 0
    stop_trading = 23

    macd_signal_df = ms.macd_rsi_signals(indicator_val_rolling, y_train['SI1:RSI'], macd_m1, macd_m2, macd_enter, macd_exit, rsi_upper, rsi_lower,
                        start_trading, stop_trading, plot = False, plot_start = start)
    init_cap = 1_000_000
    stoploss = .1
    transaction_cost = .0002
    brokerage_cost = 0.001
    costs = transaction_cost + brokerage_cost
    slippage_max = slip_train['WindowMax']
    slippage_min = slip_train['WindowMin']

    backTest = BT(init_cap, y_train['SI1:close'], macd_signal_df, stoploss, lot_size, costs, slippage_min, slippage_max)
    backTest.run_backtest(plot=False)
    strat_tot_ret = (backTest.PnL/init_cap).iloc[-1].values[0]
    time_scale = (backTest.PnL.index[-1] - backTest.PnL.index[0]).days/365
    strat_vol = (backTest.PnL/init_cap).std().values[0]
    ret_scaled = strat_tot_ret/time_scale
    vol_scaled = strat_vol / np.sqrt(time_scale)
    sharpe_ratio = ret_scaled/vol_scaled
    min_out = -backTest.PnL.iloc[-1].values[0]
    return min_out

def optimized_params(df, y_train, ind_val_rol,slip_train, x0, max_iter, lot_size):
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[3] - x[2] + 0.06},
            {'type': 'ineq', 'fun': lambda x: x[5] - x[4]},
            {'type': 'eq', 'fun': lambda x: x[5] + x[4] - 100})
    bnds = [(1,12),(5,40),(0.05,0.3),(0.01,0.1),(55,90),(10,45)]
    t1 = time.time()
    res = minimize(optimize_TS, x0, args = (df, y_train, ind_val_rol, slip_train, lot_size), method='SLSQP', bounds=bnds,
                   constraints=cons, options={'maxiter': max_iter, 'disp': True})
    print(time.time() - t1)
    return(res.x)


def benchmark_plot(bench, ret):
    # Plot Returns
    df = bench.copy()
    df = df[df.index.isin(ret.index)]
    sns.set_style("whitegrid")
    df['TS'] = ret
    df['TS'].fillna(0)
    (df + 1).cumprod().plot(figsize = (16,6))
    plt.title("Returns vs Benchmark Index", fontsize = 18)
    plt.ylabel("Cumulative Returns")
    save_fig('benchmark_plot')
    plt.show();
    return

def benchmark_corr(bench, ret):
    df = bench.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    df['TS'].fillna(0)
    return df[['GSCI','SI1','TS']].corr()

def benchmark_regr_plots(bench, ret):
    # Plot
    df = bench.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.regplot(x=df['TS'], y=df['GSCI'], color='blue', marker='+', ax = axs[0])
    axs[0].title.set_text('TS vs GSCI Returns')
    axs[0].set_xlabel('Daily TS Returns')
    axs[0].set_ylabel('GSCI Returns')

    sns.regplot(x=df['TS'], y=df['SI1'], color='magenta', marker='+', ax = axs[1])
    axs[1].title.set_text('TS vs Silver Returns')
    axs[1].set_xlabel('Daily TS Returns')
    axs[1].set_ylabel('Silver Returns')

    plt.tight_layout()
    save_fig('bench_regr_plots')
    plt.show();
    return

def benchmark_OLS(bench, ret):
    df = bench.copy()
    df = df[df.index.isin(ret.index)]
    df['TS'] = ret
    factors = df[['GSCI', 'SI1']]
    rhs = sm.add_constant(factors)
    lhs = df['TS']
    res = sm.OLS(lhs, rhs, missing='drop').fit()
    display(res.summary())
    return

def benchmark_analysis(bench, ret):
    benchmark_plot(bench, ret)
    benchmark_regr_plots(bench, ret)
    benchmark_OLS(bench, ret)
    res = benchmark_corr(bench, ret)
    display(res)

    return

def drawdown_plot(ret):
    plt.figure(figsize=(16,10))
    i = np.argmax(np.maximum.accumulate(ret) - ret) # end of the period
    j = np.argmax(ret[:i]) # start of period

    plt.plot(ret)

    drawdown_start = ret.index[j]
    drawdown_end = ret.index[i]

    drawdown_peak = ret.iloc[j]
    drawdown_min = ret.iloc[i]

    drawdown = (drawdown_peak - drawdown_min)/drawdown_peak

    plt.scatter(drawdown_start,drawdown_peak, marker='o',color='red',label = 'Peak')
    plt.scatter(drawdown_end,drawdown_min, marker='x',color='red',label = 'Min')

    date_range = [drawdown_start, drawdown_end]
    data_range = [drawdown_peak, drawdown_min]

    plt.plot(date_range, data_range, '--', color = 'r',label = 'Max Drawdown: ' + str(round(100*drawdown,2))+'%')

    i = np.argmax(ret - np.minimum.accumulate(ret)) # end of the period
    j = np.argmin(ret[:i]) # start of period

    upside_start = ret.index[j]
    upside_end = ret.index[i]

    upside_peak = ret.iloc[i]
    upside_min = ret.iloc[j]

    upside = (upside_peak - upside_min)/upside_min
    plt.scatter(upside_start,upside_min, marker='o',color='green',label = 'Min')
    plt.scatter(upside_end,upside_peak, marker='x',color='green',label = 'Peak')

    date_range = [upside_start, upside_end]
    data_range = [upside_min, upside_peak]

    plt.plot(date_range, data_range, '--', color ='green', label = 'Max Upside: ' + str(round(100*upside,2))+'%')

    plt.title('Max Drawdown and Upside PnL', size = 18)
    plt.ylabel('Cumulative Returns', size = 16)
    plt.xlabel('Date', size = 16)
    plt.legend(fontsize = 'large')
    plt.plot()
    save_fig('drawdown_plot')
    plt.show()
    return


def kde_distribution_plot(ret):
    n_x = np.arange(-3,3,0.001)
    y = pd.Series(norm.pdf(n_x,0,1), index = n_x, name = 'Normal Distribution')
    fig, ax = plt.subplots(1, figsize=(10,10))
    plt.style.use('fivethirtyeight')
    fig.suptitle('Trading Strategy Kernel Density Plot vs Normal Distribution', fontsize = 20)

    data = ret
    mean = data.mean()
    std = data.std()
    normalize = (data - mean)/std

    ax.plot(n_x,y, c= 'r', lw=3, label = 'Normal Distribution')
    ax.set_ylabel('Density')
    normalize.plot.kde(ax=ax, label = 'Trading Strategy', lw=3)
    ax.legend(loc="upper right", fontsize = 14)
    save_fig('kde_plot')
    plt.show()
    plt.style.use('seaborn')
    return


def distribution_plots(ret, plot = True):
    pos_ret = ret[ret['TS']>0]['TS']
    neg_ret = ret[ret['TS']<0]['TS']
    mean_pos = pos_ret.mean()
    mean_neg = neg_ret.mean()
    count_pos = len(pos_ret)
    count_neg = len(neg_ret)
    if plot:
        fig, ax = plt.subplots(1,2,figsize=(18,10))
        ax[0].boxplot([100*pos_ret,100*neg_ret], widths=0.5, patch_artist=True,
                        showmeans=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 1},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                  "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})

        x_ticks_labels = ['Winning Trades','Losing Trades']
        ax[0].set_title("Distribution of Returns for Winning and Losing Trades", fontsize = 18)
        ax[0].set_ylabel("Return in %", fontsize = 14)
        ax[0].set_xticklabels(x_ticks_labels, fontsize=14)

        ax[1].bar([1,2],[count_pos,count_neg], width=0.3, edgecolor="white", linewidth=0.7, align = 'center', tick_label = [-1,1])
        ax[1].set_title("Distribution of Number of Winning vs Losing Trades", fontsize = 18)
        ax[1].set_ylabel('Count', fontsize = 14)
        ax[1].set_xticklabels(x_ticks_labels, fontsize=14)
        save_fig('dist_plots')
        plt.show()
    return mean_pos, mean_neg, count_pos, count_neg

def acf_pacf_plots(ret):
    fig, ax = plt.subplots(2,1, figsize = (12,12))
    plt.suptitle("Autocorrelation and Partial Autocorrelation of the Trading Stategy")
    sm.graphics.tsa.plot_acf(ret['TS'], ax = ax[0], lags = 14)
    sm.graphics.tsa.plot_pacf(ret['TS'], ax = ax[1], lags = 14)
    save_fig('acf_pacf')
    plt.show()
    res = adfuller(ret['TS'], autolag='AIC')
    print(f'ADF Statistic: {res[0]}')
    print(f'p-value: {res[1]}')
    for key, value in res[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    return

def seasonality_plot(ret):
    seasonality = sm.tsa.seasonal_decompose(ret['TS'], model='additive', period=20)
    plt.rcParams['figure.figsize'] = [16,10]
    seasonality.plot()
    save_fig('seasonality')
    plt.show()
    return

def qq_plot(ret):
    z_ret = (ret['TS'] - ret['TS'].mean())/(ret['TS'].std())
    fig = sm.qqplot(z_ret,line="45")
    fig.suptitle("QQ-Plot for the Exponential-Regression")
    save_fig('QQ_plot')
    plt.show()
    return

def compute_kelly_fraction(ret):
    dist_stats = distribution_plots(ret, plot = False)
    w = dist_stats[0]*100
    l = abs(dist_stats[1])*100
    p = ((dist_stats[2]+1)/(dist_stats[2]+dist_stats[3]+2))
    f = ((p*w)-(1-p)*l)/(w*l)
    return f

if __name__ == '__main__':
    print("Hello There")
