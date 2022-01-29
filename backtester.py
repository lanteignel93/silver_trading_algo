import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

class BackTest:
    def __init__(self, N, df1, signals, stop_loss, lotsize, tc, slip_min, slip_max):
        self.initial_capital = N
        self.df1 = df1
        self.signals = signals
        self.stop_loss = stop_loss
        self.lot_size = lotsize
        self.scaled_pos = pd.DataFrame(index =df1.index)

        # Slippage Parameters
        self.price_min = slip_min
        self.price_max = slip_max
        self.slippage_cost = 0.0

        self.pfl = pd.DataFrame(columns = ['Holdings'], index = df1.index)
        self.price1 = df1
        self.PnL = pd.DataFrame(columns = ['PnL'],index = df1.index)
        self.Total = 0.0
        self.num_stop_loss = 0
        self.trade_total = pd.DataFrame(columns = ['Trade_Total'],index = df1.index)

        # Identify when to buy/sell
        self.buy_idx = np.where(self.signals['signal'] == 1)
        self.sell_idx = np.where(self.signals['signal'] == -1)

        # Identify when to enter trade
        self.trade_enter = np.logical_and(self.signals['signal'] != 0, self.signals['positions'] != 0)
        self.trade_enter_buy = np.logical_and(self.signals['signal'] == 1, self.signals['positions'] != 0)
        self.trade_enter_sell = np.logical_and(self.signals['signal'] == -1, self.signals['positions'] != 0)

        # Identify when to take profit
        self.profit_take_idx = np.where(abs(self.signals['positions']) > 0)
        self.num_trades = len(self.profit_take_idx[0])/2 # Divide by 2 since signals[positions] includes the purchase AND sale
        # Calculate transaction costs = basis points * #trades * notional (equal notional traded each time)
        self.transaction_costs = tc*self.num_trades*2*self.initial_capital*self.lot_size

    def trade(self):
        # Scale Positions to Trade Equal Notional
        self.scale_positions()
        # Calculate cost/premium when entering trade
        ### Slippage - If signal = 1 -> Buy at the Max | Signal = -1 -> Sell at the Min
        enter_buy = self.signals.loc[self.trade_enter_buy,'signal'].values*(self.scaled_pos.loc[self.trade_enter_buy, 'Size'] * self.price_max.loc[self.trade_enter_buy])
        enter_sell = self.signals.loc[self.trade_enter_sell,'signal'].values*(self.scaled_pos.loc[self.trade_enter_sell, 'Size'] * self.price_min.loc[self.trade_enter_sell])
        enter_no_slip = self.signals.loc[self.trade_enter, 'signal'].values*(self.scaled_pos.loc[self.trade_enter, 'Size'] * self.price1.loc[self.trade_enter])

        # Update total slippage cost
        self.slippage_cost = abs(enter_no_slip.loc[self.trade_enter_buy] - enter_buy).sum() + abs(enter_no_slip.loc[self.trade_enter_sell] - enter_sell).sum()

        self.trade_total.loc[self.trade_enter_buy, 'Trade_Total'] = enter_buy
        self.trade_total.loc[self.trade_enter_sell, 'Trade_Total'] = enter_sell
        self.trade_total = self.trade_total.fillna(method = 'ffill')

        # Update holdings when holding position
        self.pfl['Holdings'] = self.signals.loc[:, 'signal'].values*(self.scaled_pos.loc[:, 'Size'] * self.price1.loc[:])

        self.pfl['Cur_PnL'] = self.pfl['Holdings'] - self.trade_total['Trade_Total']

        # If there is a stop Loss - Take loss and then stop trading till next "new" signal
        self.check_stopLoss()

        # Update PnL
        temp_pnl = self.signals['signal'][self.profit_take_idx[0]-1].values*(self.scaled_pos.iloc[self.profit_take_idx[0]]['Size'].values*self.price1.iloc[self.profit_take_idx[0]])
        self.PnL['PnL'][self.profit_take_idx[0]] = temp_pnl + self.pfl['Cur_PnL'][self.profit_take_idx[0]-1].values - self.trade_total['Trade_Total'][self.profit_take_idx[0]].values

        # Update Total Holdings
        self.Total = self.PnL.fillna(0).cumsum() + self.initial_capital - self.transaction_costs

        return


    def scale_positions(self):
        notional = self.initial_capital*self.lot_size
        self.scaled_pos['Size'] = (notional / (self.price1.loc[self.trade_enter]))
        self.scaled_pos['Size']= self.scaled_pos.fillna(method = 'ffill').replace(np.inf,0)
        self.scaled_pos['Size'] = self.scaled_pos.round(0)

        return

    def check_stopLoss(self):
        stop_loss_idx = np.where(self.pfl['Cur_PnL'] < -1*self.stop_loss*self.initial_capital)
        for i in stop_loss_idx[0]:
            if i+1 >= len(self.pfl):
                break
            stop_date = self.pfl.iloc[i+1].name
            if self.pfl.iloc[i+1, 0] == 0: # Check if holdings are already zero'ed out
                continue
            else:
                next_trade = np.where(stop_date < self.trade_enter[self.trade_enter].index)
                if len(next_trade[0]) != 0:
                    for day in range(len(next_trade[0])):
                        next_trade_date = self.trade_enter[self.trade_enter].index[next_trade[0][day]]
                        # Adjust Date to 1 day prior
                        next_trade_date = str(pd.to_datetime(next_trade_date) - datetime.timedelta(days = 1))
                        self.pfl[stop_date:next_trade_date] = 0
                        self.PnL.loc[stop_date, 'PnL'] = self.pfl.iloc[i, -1]
                        self.num_stop_loss += 1
                        break

        return

    def update_pfl(self):
        self.pfl = self.pfl.fillna(0)
        self.PnL = self.PnL.fillna(0).cumsum()
        return


    def calc_results(self):
        ret_cap = (self.PnL.iloc[-1][0] + self.initial_capital) / self.initial_capital - 1
        stats = {"Trades": self.num_trades,
                 "PnL": self.PnL.iloc[-1][0],
                 "Return_on_Capital": ret_cap,
                 "StopLoss": self.num_stop_loss,
                "Transaction Costs": self.transaction_costs,
                "Slippage Costs": self.slippage_cost}
        return stats

    def plot(self):
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16,4))
        ax1 = fig.add_subplot(111, ylabel='PnL')
        self.PnL.plot(ax=ax1, lw=2.)
        plt.title("Portfolio Back Test Results", fontsize = 18)
        save_fig('TS_bt_res')
        plt.show()

        return

    def plot_entry_exit_points(self):
        fig, ax = plt.subplots(1, figsize = (20,10))
        x_sell = self.trade_enter_sell[self.trade_enter_sell == True].index
        x_buy = self.trade_enter_buy[self.trade_enter_buy == True].index
        self.df1.plot(c = 'k',ax=ax,lw = 3)
        ax.plot(x_buy, self.df1[self.df1.index.isin(x_buy)], '^',markersize = 10, c='g', label='Buy')
        ax.plot(x_sell, self.df1[self.df1.index.isin(x_sell)], 'v',markersize = 10, c='r', label='Sell')
        ax.set_title('Silver Price with Strategy Buy/Sell Points', fontsize = 18)
        ax.set_ylabel('Silver $', fontsize = 14)
        ax.set_xlabel('Date', fontsize = 14)
        ax.legend(fontsize = 15)
        save_fig('entry_exit')
        plt.show()


    def run_backtest(self, plot = True):
        self.trade()
        self.update_pfl()
        stats = self.calc_results()
        if plot:
            self.plot()
        return stats
