import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tseries.offsets import BDay

import backtest_tools.portfolio_tools as backtest_tools
import backtest_tools.data_cleaning_tools as cleaning

class factor_strategy():

    def __init__(self, returns: pd.DataFrame, factor_function, 
                                              quantile=10, 
                                              rolling_window=20, 
                                              rebal_freq = 'M', 
                                              factor_type='long', 
                                              p=0.50, 
                                              filter_type='high_vol', 
                                              daily_ret_filter_threshold=10):
        """ Class to compose factor strategies based on the returns of given securities.

        Args:
            returns (pd.DataFrame): _description_
            factor_function (_type_): _description_
            quantile (int, optional): _description_. Defaults to 10.
            rolling_window (int, optional): _description_. Defaults to 20.
            rebal_freq (str, optional): _description_. Defaults to 'M'.
            factor_type (str, optional): _description_. Defaults to 'long'.
            p (float, optional): _description_. Defaults to 0.50.
            filter_type (str, optional): _description_. Defaults to 'high_vol'.
            daily_ret_filter_threshold (int, optional): _description_. Defaults to 10.

        Raises:
            ValueError: _description_
        """
         
        # Function to compute feature/factor data
        self.factor_function = factor_function

        # Quantile to long and short (e.g., 10th percentile and 90th percentile)
        self.quantile = quantile/100

        # Rolling length of time over which factor is computing (e.g., 20 days = 1 month)
        self.rolling_window = rolling_window

        # Use either pandas shorthand for rebal_freq or numeric daily value using BDay
        if type(rebal_freq) is str:
            self.rebal_freq = rebal_freq
        else:
            self.rebal_freq = BDay(int(rebal_freq))
            raise ValueError('Numeric rebalancing frequencies are currently unsupported.')
        
        # Determines whether to long or short top quantile of the given factor
        self.factor_type = factor_type

        # Determines the proportion of data to drop when filtering for volatility
        self.p = p

        # Determines the direction of volatitly to favor when filtering for volatility (low vol or high vol)
        self.filter_type = filter_type

        # Determines the max allowable daily return before that security is dropped
        self.daily_ret_filter_threshold = daily_ret_filter_threshold

        # Clean returns (drop outlier daily returns + filter for volatilty)
        self.returns = self.clean_returns(returns=returns, p=self.p, filter_type=self.filter_type, daily_ret_threshold=self.daily_ret_filter_threshold)
        
        # Compute factor
        self.full_factor_history = self.get_factor(returns=self.returns, factor_function=self.factor_function)
        
        # Get factor values at a rebal_freq period frequency
        self.factor = self.full_factor_history.groupby(pd.Grouper(freq=self.rebal_freq)).last()

        # Get portfolio weights and factor returns
        self.portfolio_weights, self.factor_returns = self.get_weights_and_returns(factor=self.factor, returns=self.returns, quantile=self.quantile, factor_type=self.factor_type)

        # Execute performance analysis
        self.performance_summary = backtest_tools.performance_summary(self.factor_returns)

        # Get cumulative returns
        self.cumulative_returns = backtest_tools.cumulative_returns(self.factor_returns)

        # Plot cumulative returns for rapid ideation
        self.cumulative_returns.plot(figsize=(12, 6))
        plt.show()

        return
    
    def get_factor(self, returns, factor_function):
        """ Custom Factor Function.

        Args:
            returns (_type_): _description_
            factor_function (_type_): _description_

        Returns:
            _type_: _description_
        """

        return factor_function(returns)

    def clean_returns(self, returns: pd.DataFrame, p=0.50, filter_type='high_vol', daily_ret_threshold=10):
        """ 1) Drops outlier daily returns at a user-specified max return threshold.
            2) Filters for a user-specified proportion of assets with high/low volatility.

        Args:
            returns (pd.DataFrame): _description_
            p (float, optional): _description_. Defaults to 0.50.
            filter_type (str, optional): _description_. Defaults to 'high_vol'.
            daily_ret_threshold (int, optional): _description_. Defaults to 10.

        Returns:
            _type_: Filtered returns.
        """
        returns = cleaning.drop_outlier_returns(returns, daily_ret_threshold=daily_ret_threshold)
        returns = cleaning.volatility_filter(returns, p=p, filter_type=filter_type)

        return returns

    # Function to assign portfolio weights to ranked tickers
    def get_weights(self, factor: pd.Series, quantile: float, factor_type='long'):
        """_summary_

        Args:
            factor (pd.Series): _description_
            factor_type (str, optional): _description_. Defaults to 'long'.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        n = int(len(factor.dropna())*quantile)

        if n>0:

            try:

                # Get quantile largers tickers' cum rets
                nlargest = factor.nlargest(n)
                nsmallest = factor.nsmallest(n)

                # print('='*50)
                # print(nlargest.index)
                # print(factor.nlargest(n))
                # print(nsmallest.index)
                # print(factor.nsmallest(n))
                # print('='*50)

                long_weight = 1/(n*2)
                short_weight = 1/(n*2)*-1

                if(factor_type=='long'):
                    nlargest[nlargest.index] = long_weight  
                    nsmallest[nsmallest.index] = short_weight
                elif(factor_type=='short'):
                    nlargest[nlargest.index] = short_weight 
                    nsmallest[nsmallest.index] = long_weight
                else:
                    raise ValueError('Respecify factor_type')

                weights = pd.concat([nlargest, nsmallest])

                return weights
            
            except: 
                print(n, factor.nlargest(n))
                
        return factor*0

    def get_weights_and_returns(self, factor: pd.DataFrame, returns: pd.DataFrame, quantile: float, factor_type='long'):

        wts = {}

        for i, tmp_factor in factor.T.items():
            wts[i] = self.get_weights(tmp_factor, quantile=quantile, factor_type=factor_type)
        
        # Tranpose pd.DataFrame to ensure indices are dates and columns are tickers
        wts = pd.DataFrame(wts).T

        # Fill NaNs to 0 so we can ffill the weights' pd.DataFrame
        wts = wts.fillna(0)
        # Create a temporary pd.DataFrame with returns.index which will enable us to ffill weights as well
        tmp_wts = pd.DataFrame(index=returns.index)
        wts = pd.concat([wts, tmp_wts], axis=1).ffill()
        wts = wts.loc[returns.index, :]

        # Check that weights add to 0 to ensure market netural strategy
        wts_sum = wts.sum().sum().round(4)
        print(f"Weights SUM: {'Market Neutral' if wts_sum == 0 else 'Not Market Netrual'} = {wts_sum}")

        # Plot example of stock weights over time for given factor
        wts.iloc[:, 1:20].plot(title="Sample Weights Over Time", figsize=(12, 6))
        plt.show()

        # Compute returns
        factor_returns = (wts*returns).sum(1)

        return wts, factor_returns


def expanding_z_score(factor: pd.DataFrame):
    """ Function to generate expanding Z-Scores for a given pd.DataFrame of Factor data. This facilitates truly implementable 
        backtesting of factors.

    Args:
        factor (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """

    expanding_z_score = {}

    for ticker, series in factor.iloc[:, 0:5].items():

        tmp_z_score_factor = {}    

        for date, fact in series.items():
            tmp_z_score_factor[date] = (fact - value.loc[:date].mean())/value.loc[:date].std()

        tmp_z_score_factor = pd.Series(tmp_z_score_factor, name=ticker)
        expanding_z_score[ticker] = tmp_z_score_factor  

    return pd.DataFrame(expanding_z_score)