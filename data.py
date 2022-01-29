import pandas as pd
import pickle as pickle
import os

ROOT_DIR = os.getcwd()
IMAGES_PATH = os.path.join(ROOT_DIR, "images")
DATA_PATH = os.path.join(ROOT_DIR, "data")
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)


def rename_cols(df,name):
    new_names = {}
    for i in df.columns:
        new_names[i] = (name + ":" + i)

    return new_names



if __name__ == '__main__':

    silver = pd.read_csv(DATA_PATH +'\\' + "SI1_AD_RSI12_BB20_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    silver = silver.rename(columns = rename_cols(silver,"SI1"))

    gold = pd.read_csv(DATA_PATH +'\\' + "GC1_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    gold = gold.rename(columns = rename_cols(gold,"GC1"))

    copper = pd.read_csv(DATA_PATH +'\\' + "HG1_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    copper = copper.rename(columns = rename_cols(copper,"HG1"))

    spx = pd.read_csv(DATA_PATH +'\\' + "SPX_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    spx = spx.rename(columns = rename_cols(spx,"SPX"))

    vix = pd.read_csv(DATA_PATH +'\\' + "VIX_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    vix = vix.rename(columns = rename_cols(vix,"VIX"))

    us10y = pd.read_csv(DATA_PATH +'\\' + "US10Y_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    us10y = us10y.rename(columns = rename_cols(us10y,"10Y"))

    usdjpy = pd.read_csv(DATA_PATH +'\\' + "USDJPY_Hourly_01_01_2020_11_04_2021.csv", parse_dates = True, index_col = 0)
    usdjpy = usdjpy.rename(columns = rename_cols(usdjpy,"USDJPY"))

    # Download Fama French Data. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    ff = pd.read_csv(DATA_PATH +'\\'+ "FF_daily.csv", skiprows = 3, index_col = 0)
    ff = ff.drop(index = 'Copyright 2021 Kenneth R. French')
    ff.index = pd.to_datetime(ff.index, format = '%Y%m%d')
    # Subset to Time Frame of Analysis
    ff = ff.loc['2020-01-01':,:] / 100
    ff.to_pickle(DATA_PATH + '\\ff.pkl')
    #ff.describe()

    # NOTE Need SPX & VIX To have Correct Index
    df = silver.merge(gold, how = 'left', on = 'time')
    df = df.merge(copper, how = 'left', on = 'time')
    df = df.merge(usdjpy, how = 'left', on = 'time')
    df = df.merge(us10y, how = 'left', on = 'time')
    df.to_pickle(DATA_PATH + '\\merge_df.pkl')

    # S&P GSCI Commodity Index
    gsci = pd.read_csv(DATA_PATH +'\\' +"GSCI_daily.csv", parse_dates = True,index_col = 0)
    gsci['GSCI'] = gsci['close']
    gsci = gsci.set_index(gsci.index.date)['GSCI']

    # Daily Silver Futures
    SI1 = df.between_time('17:00','17:30').set_index(df.between_time('17:00','17:30').index.date)['SI1:close']

    # Create Benchmark Dataset
    benchmark_df = pd.DataFrame()
    benchmark_df['SI1'] = SI1
    benchmark_df['GSCI'] = gsci
    bench_ret = benchmark_df.fillna(method = 'ffill').pct_change()
    bench_ret.to_pickle(DATA_PATH + '\\bench_ret.pkl')
