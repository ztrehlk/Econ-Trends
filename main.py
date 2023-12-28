import requests
import pandas as pd
import numpy as np
import functools as ft
import polars as pl
from yahoofinancials import YahooFinancials
import datetime

class FredAPI:
    def __init__(self, api_key:str):
        self.api_key = api_key

    def request(self, series_id:str):
        """Simple API request to get all the data available
        from a given `series_id`. Must have an API key set up
        to pull from this URL.

        Args:
            api_key (str): unique API key as granted from Fred API
            series_id (str): Series to pull from

        Returns:
            dict: data of date and value
        """
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.api_key}&file_type=json'
        r = requests.get(url)
        return [{'date': i['date'], 'value': i['value']} for i in r.json()['observations']]

    def series_details(self, series_id:str):
        """Find the title of a series given the series_id.

        Args:
            api_key (str)
            series_id (str)

        Returns:
            str: title
        """
        url = f'https://api.stlouisfed.org/fred/series?series_id={series_id}&api_key={self.api_key}&file_type=json'
        r = requests.get(url)
        return r.json()['seriess'][0]

    def full_dataframe(self, series_id_list:list):
        """Takes in a list of Series IDs to get Fred data and join them
        based on the date column. Also names the columns accordingly.

        Args:
            api_key (str): API Token.
            series_id_list (list): List of desired series.

        Returns:
            polars.DataFrame: Joined DataFrame of all data.
        """
        dfs = []

        for series_id in series_id_list:
            df = pl.DataFrame(self.request(series_id), schema=[("date", str), ("value", str)])\
                    .with_columns((pl.col("value").str.replace(".", 0).cast(pl.Float32)).alias("value"),\
                                    (pl.col("date").str.to_date("%Y-%m-%d")))

            series_data = self.series_details(series_id)
            title = f'{series_data["title"]} ({series_data["frequency"]})'
            df = df.rename({'value': title})
            dfs.append(df)

        df_joined = ft.reduce(lambda left, right: left.join(right, on='date', how='outer_coalesce'), dfs).sort(by="date")
        return df_joined
    
    def monthly_df(self, series_id_list:list) -> pl.DataFrame:
        """Inclusive function that calls the full_dataframe() function
        to get all the data of each series_id then for columns that record
        daily values, transforming that data into monthly representations.

        Uses the (Monthly), (Daily) names in the headers to indicate
        how the function proceeds.

        Args:
            series_id_list (list): list of series_id's

        Returns:
            pl.DataFrame
        """

        df = self.full_dataframe(series_id_list)
        new_dfs = []

        for col_name in list(df.columns):
            if "Daily" in col_name:
                new_dfs.append(Transformations.day_to_month_column(df, col_name))
            elif col_name=="date":
                pass
            else:
                new_dfs.append(df[["date", col_name]].drop_nulls())

        df_joined = ft.reduce(lambda left, right: left.join(right, on='date', how='outer_coalesce'), new_dfs).sort(by="date")

        return df_joined    
    
class Transformations:
    def day_to_month_column(df:pl.DataFrame, col_name:str) -> pl.DataFrame:
        """Takes in a polars dataframe and designated column name
        to perform the necessary row operations to summarize daily
        values into monthly representations.

        Args:
            df (pl.DataFrame): full polars dataframe
            col_name (str): daily column for transformation

        Returns:
            pl.DataFrame: grouped object with summary stats
        """
        day_col = df[["date", col_name]].sort(by="date")
        day_col = day_col.with_columns(pl.col("date").dt.strftime("%Y-%m").str.to_date("%Y-%m")).drop_nulls()
        day_col = day_col.group_by("date").agg(pl.col(col_name).first().name.prefix("first_"),
                                                pl.col(col_name).last().name.prefix("last_"),
                                                pl.col(col_name).min().name.prefix("min_"),
                                                pl.col(col_name).max().name.prefix("max_"),
                                                pl.col(col_name).mean().name.prefix("mean_"),
                                                pl.col(col_name).median().name.prefix("median_"),
                                                pl.col(col_name).sum().name.prefix("sum_"),
                                                pl.col(col_name).quantile(.25).name.prefix("quantile_25_"),
                                                pl.col(col_name).quantile(.75).name.prefix("quantile_75_"),
                                                ).sort("date")
        return day_col
    

class YahooFinanceAPI:
    def get_prices_daily(self, ticker:str) -> list:
        '''
        Uses YahooFinance to get daily stock history information on
        a given company. This grants the raw set which can be 
        expanded on using feature engineering techniques.

        Parameters:
            - ticker (string): stocker ticker symbol

        Returns:
            - df
        '''
        yahoo_financials = YahooFinancials(ticker)
        data = yahoo_financials.get_historical_price_data('1900-01-01', str(datetime.date.today()), 'daily')[ticker]["prices"]
        return data

    def daily_stock_dataframe(self, ticker:str) -> pl.DataFrame:
        """Transforms yahoo finance data into polars dataframe.
        Uses get_prices_daily to extract data then performs basic 
        transformations and minor calculations:
        - High-Low
        - Open-Close

        Args:
            ticker (str): stock ticker

        Returns:
            pl.DataFrame
        """
        daily_stock_df = pl.DataFrame(self.get_prices_daily(ticker)).with_columns(pl.col("formatted_date").str.to_date("%Y-%m-%d"))
        daily_stock_df = daily_stock_df.with_columns((pl.col("high")-pl.col("low")).alias("H-L"), 
                                                    (pl.col("close")-pl.col("open")).alias("C-O"))\
                                        .drop("date")\
                                        .rename({"formatted_date": "date"})\
                                        .select([pl.col("date"), pl.col("volume"), 
                                                 pl.col("open"), pl.col("close"), 
                                                 pl.col("high"), pl.col("low"), 
                                                 pl.col("H-L"), pl.col("C-O")])\
                                        .sort(by="date")
        return daily_stock_df



    def monthly_stock_dataframe(self, ticker:str) -> pl.DataFrame:
        """Uses daily_stock_df to get initial daily dataframe. Then uses
        day_to_month_col to convert each column into its own transformed, 
        monthly dataframe which are then all joined together to produce a 
        monthly_stock_df.

        Args:
            ticker (str): stock ticker

        Returns:
            pl.DataFrame
        """
        daily_df = self.daily_stock_dataframe(ticker)
        val_columns = [i for i in daily_df.columns if i!="date"]

        dfs = []
        for col in val_columns:
            dfs.append(Transformations.day_to_month_column(daily_df, col))

        monthly_stock_df = ft.reduce(lambda left, right: left.join(right, on='date', how='outer_coalesce'), dfs).sort(by="date")
        return monthly_stock_df

# Example usage:
# fred_api = FredAPI(api_key="your_fred_api_key")
# yahoo_finance_api = YahooFinanceAPI()

# series_id_list = ["series_id_1", "series_id_2"]
# fred_data = fred_api.full_dataframe(series_id_list)

# ticker = "AAPL"
# stock_data_daily = yahoo_finance_api.daily_stock_dataframe(ticker)
# stock_data_monthly = yahoo_finance_api.monthly_stock_dataframe(ticker)
