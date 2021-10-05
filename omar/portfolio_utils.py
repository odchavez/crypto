import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime

def date_cleaner(date_series, format = "%Y-%m-%d"):
    empty_list = [None]*len(date_series)
    for i in range(len(empty_list)):
        empty_list[i] = datetime.datetime.strptime(date_series[i][:10], format)
    return empty_list


class Portfolio:
    
    def __init__(self, data, symbols, weights, initial_investment, start_date, end_date, rebalance_dates):

        self.data=data
        self.wts = weights
        self.initial_investment = initial_investment
        self.symbols = symbols
        
        self.portfolio_rebalance_datetime = np.array([datetime.datetime.strptime(item, "%Y-%m-%d") for item in rebalance_dates])
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.endp1_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        #get valid rebalance dates
        valid_rebalance_datetime = np.array(
            [rb_date for rb_date in self.portfolio_rebalance_datetime if self.start_date<rb_date and rb_date < self.endp1_date]
        )
        self.event_datetimes = np.insert(valid_rebalance_datetime, 0, self.start_date)
        self.event_datetimes = np.append(self.event_datetimes, self.endp1_date)
        
    def period_return(self, returns_df, weights=None):
        n_dates = returns_df.shape[0]
        daily_portfolio_returns = np.zeros((n_dates,returns_df.shape[1]))
        if weights is None:
            current = np.ones(returns_df.shape[1]) # we start with 100% of funds invested
        else:
            current = weights
            
        for i in range(n_dates):
            gorc = (1+returns_df.iloc[i,:]) #gorc "growth or contraction"
            current *= gorc
            daily_portfolio_returns[i,:]=current
        return daily_portfolio_returns
    
    def estimate_portfolio_returns(self):
    
        self.daily_portfolio_values = np.ones((1,len(self.wts)))
        self.daily_portfolio_values[0,:] = self.initial_investment*self.wts

        for i in range(len(self.event_datetimes)-1):
            sub_start = self.data.date>=self.event_datetimes[i]
            sub_end = self.data.date<self.event_datetimes[i+1]
            sub_index = np.logical_and(sub_start,sub_end)
            
            portfolio_values = self.period_return(
                returns_df=self.data[self.symbols][sub_index], 
                weights=np.sum(self.daily_portfolio_values[-1,:])*self.wts)
            
            self.daily_portfolio_values = np.vstack((self.daily_portfolio_values, portfolio_values))
            
            self.total_portfolio_value = np.sum(self.daily_portfolio_values[-1,:])
            
    def get_sharpe_ratio(self):
        vals = np.sum(self.daily_portfolio_values, axis=1)
        daily_mean = np.mean(vals)
        daily_std  = np.std(vals)
        SR = np.sqrt(365)*daily_mean/daily_std
        return SR
    
    def get_sharpe_report(self):
        portfolio_SR = self.get_sharpe_ratio()
        daily_mean = np.mean(self.daily_portfolio_values, axis=0)
        daily_std  = np.std(self.daily_portfolio_values, axis=0)
        individual_SR = np.sqrt(365)*daily_mean/daily_std
        df = pd.DataFrame(columns=list(self.symbols))
        df.loc[0] = individual_SR
        df['Portfolio SR'] = self.get_sharpe_ratio()
        return df
    
class get_history:
    def __init__():
        self.cnames=[
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore"
        ]
        self. dates = [
            "2018-01",
            "2018-02",
            "2018-03",
            "2018-04",
            "2018-05",
            "2018-06",
            "2018-07",
            "2018-08",
            "2018-09",
            "2018-10",
            "2018-11",
            "2018-12",
            "2019-01",
            "2019-02",
            "2019-03",
            "2019-04",
            "2019-05",
            "2019-06",
            "2019-07",
            "2019-08",
            "2019-09",
            "2019-10",
            "2019-11",
            "2019-12",
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
            "2020-05",
            "2020-06",
            "2020-07",
            "2020-08",
            "2020-09",
            "2020-10",
            "2020-11",
            "2020-12",
            "2021-01",
            "2021-02",
            "2021-03",
            "2021-04",
            "2021-05",
            "2021-06",
            "2021-07",
            "2021-08",
            #"2021-09",
            #"2021-10",
            #"2021-11",
            #"2021-12",
        ]
    def get_history(self, symbol):
        
        for i in range(len(self.dates))
            path = "klines_data/"+symbol+self.dates[i]+".csv
            if i==0:
                data = pd.read_csv(path)
            else:
                temp = pd.read_csv(path)
                data = pd.concat([data,temp], ignore_index=True)
            return data
    