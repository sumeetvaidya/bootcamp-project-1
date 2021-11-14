#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import datetime
#import matplotlib.pyplot as plt
#%matplotlib inline
#import dash
#import dash_core_components as dcc
#from dash import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.graph_objs as go
#import ipywidgets as widgets
#import dash_table
#import dash_table.FormatTemplate as FormatTemplate
#from dash_table.Format import Format, Scheme, Symbol, Group

from pandas_datareader import data as pdr
import yfinance as yf
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder


from MCForecastTools import MCSimulation

__author__ = "Sumeet Vaidya, Pat Beeson, Scott Oziros and William Alford"
__copyright__ = "Copyright 2021, The Bootcamp Project"
__credits__ = ["Sumeet Vaidya", "Pat Beeson", "Scott Oziros",
                    "William Alford"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Sumeet Vaidya"
__email__ = "esumeet@gmail.com"
__status__ = "Production"


@st.cache(allow_output_mutation=True)

#Get the Holdings File
def get_holdings():
    df =  pd.read_excel('Holdings.xlsx', sheet_name='Portfolio', parse_dates=True)
    df['Acquisition Date'] = pd.to_datetime(df['Acquisition Date']).dt.normalize()
    df['Start of Year'] = pd.to_datetime(df['Start of Year']).dt.normalize()
    return df

#Record Portfolio Tickers
def get_portfolio_tickers(df):
    portfolio_tickers =df['Ticker'].to_list().copy()
    return portfolio_tickers

def get_tickers(df):
    tickers = df['Ticker'].to_list().copy()
    spy_ticker='^GSPC'
    tickers.append(spy_ticker)
    return tickers

def get_YTD_date(df):
    end_last_year=datetime.date.fromisoformat(df['Start of Year'][0].strftime('%Y-%m-%d'))
    return end_last_year


def get_ticker_data(portfolio_df, tickers):
    
    start_date = "2019-01-01"
    end_date = "2021-12-31"
    
    
    yf.pdr_override()
    df = pdr.get_data_yahoo(tickers, start=start_date , end=end_date)
    return df

def get_closing_prices(df_ticker, tickers):
    # Create and empty DataFrame for closing prices
    df_closing_prices = pd.DataFrame()
    for ticker in tickers:
        df_closing_prices[ticker]=df_ticker['Adj Close'][ticker]
    
    # Drop the time component of the date
    df_closing_prices.index = df_closing_prices.index.date
    df_closing_prices.index.name='Date'
    return df_closing_prices

# Grab the latest stock close price by using the last line in the dataframe
def get_latest_closes(df_closing_prices):
    df_close_latest = df_closing_prices.iloc[-1].to_frame('Last Close')
    df_close_latest.index.name='Ticker'
    return df_close_latest


def get_YTD_closes(df_closing_prices,end_last_year):
    #get the End of Last year Closing price using the information from the Holdings file.
    #the code should be enhanced to use the last date of the prior year
    df_close_YTD_start = df_closing_prices.loc[end_last_year].to_frame('YTD Close')
    df_close_YTD_start.index.name="Ticker"
    return df_close_YTD_start
 
def get_portfolio_closes(portfolio_df,df_closing_prices,portfolio_tickers):
    #Transpose the portfolio df to get info by Ticker
    df_T = portfolio_df.T.copy()
    df_portfolio_value = df_closing_prices.copy()
    for ticker in portfolio_tickers:
        df_portfolio_value[ticker]=df_portfolio_value[ticker] * df_T[ticker]['Quantity']
    df_portfolio_value = df_portfolio_value.drop(columns='^GSPC')
        
    return df_portfolio_value
 
def merge_portfolio_with_closes(portfolio_df, df_close_latest):
    # Merge the portfolio dataframe with the  close dataframe; they are being joined by their indexes.
    merged_portfolio = pd.merge(portfolio_df, df_close_latest, left_index=True, right_index=True)
    merged_portfolio.index.name="Ticker"
    return merged_portfolio

def calculations_with_last_close(merged_portfolio):
     #The below creates a new column which is the ticker return; takes the latest adjusted close for each position
    # and divides that by the initial share cost.
    merged_portfolio['Ticker Return'] = merged_portfolio['Last Close'] / merged_portfolio['Unit Cost'] - 1
    return merged_portfolio
    
def get_SPY_closing_prices(df_closing_prices):
    #Get S&P Closes and index by date
    df_spy_closing_prices=df_closing_prices['^GSPC'].to_frame('SPY Closes')
    df_spy_closing_prices.index = df_spy_closing_prices.index.astype('datetime64[ns]')
    df_spy_closing_prices.index.name='Date'
    return df_spy_closing_prices

def merge_portfolio_with_SPY(merged_portfolio, df_spy_closing_prices):
    # Here we are merging the new dataframe with the sp500 adjusted closes since the sp start price based on 
    # each ticker's acquisition date and sp500 close date.
    merged_portfolio_sp = pd.merge(merged_portfolio, df_spy_closing_prices, left_on='Acquisition Date', right_index=True)
    return merged_portfolio_sp

def calculations_with_SPY(merged_portfolio_sp,df_closing_prices):
    # This new column determines what SP 500 equivalent purchase would have been at purchase date of stock.
    merged_portfolio_sp['Equiv SPY Shares'] = merged_portfolio_sp['Cost Basis'] / merged_portfolio_sp['SPY Closes']
    # We are joining the developing dataframe with the sp500 closes again, this time with the latest close for SP.
    merged_portfolio_sp['SPY Latest Close'] =  df_closing_prices.iloc[-1]['^GSPC']
    
    # Percent return of SP from acquisition date of position through latest trading day.
    merged_portfolio_sp['SP Return'] = merged_portfolio_sp['SPY Latest Close'] / merged_portfolio_sp['SPY Closes'] - 1

    # This is a new column which takes the tickers return and subtracts the sp 500 equivalent range return.
    merged_portfolio_sp['Abs. Return Compare'] = merged_portfolio_sp['Ticker Return'] - merged_portfolio_sp['SP Return']

    # This is a new column where we calculate the ticker's share value by multiplying the original quantity by the latest close.
    merged_portfolio_sp['Ticker Share Value'] = merged_portfolio_sp['Quantity'] * merged_portfolio_sp['Last Close']

    # We calculate the equivalent SP 500 Value if we take the original SP shares * the latest SP 500 share price.
    merged_portfolio_sp['SPY Value'] = merged_portfolio_sp['Equiv SPY Shares'] * merged_portfolio_sp['SPY Latest Close']

    # This is a new column where we take the current market value for the shares and subtract the SP 500 value.
    merged_portfolio_sp['Abs Value Compare'] = merged_portfolio_sp['Ticker Share Value'] - merged_portfolio_sp['SPY Value']

    # This column calculates profit / loss for stock position.
    merged_portfolio_sp['Stock Gain / (Loss)'] = merged_portfolio_sp['Ticker Share Value'] - merged_portfolio_sp['Cost Basis']

    # This column calculates profit / loss for SP 500.
    merged_portfolio_sp['SPY Gain / (Loss)'] = merged_portfolio_sp['SPY Value'] - merged_portfolio_sp['Cost Basis']

    return merged_portfolio_sp

def merge_portfolio_with_YTD(merged_portfolio_sp,df_close_YTD_start):
    # Merge the overall dataframe with the adj close start of year dataframe for YTD tracking of tickers.
    # Should not need to do the outer join;

    merged_portfolio_sp_YTD = pd.merge(merged_portfolio_sp, df_close_YTD_start, left_index=True, right_index=True)
    return merged_portfolio_sp_YTD

def calculate_YTD(merged_portfolio_sp_YTD, df_closing_prices, end_last_year):
    merged_portfolio_sp_YTD['SPY YTD Close'] =  df_closing_prices.loc[end_last_year]['^GSPC']
    # YTD return for portfolio position.
    merged_portfolio_sp_YTD['Share YTD'] = merged_portfolio_sp_YTD['Last Close'] / merged_portfolio_sp_YTD['YTD Close'] - 1

    # YTD return for SP to run compares.
    merged_portfolio_sp_YTD['SPY YTD'] = merged_portfolio_sp_YTD['SPY Latest Close'] / merged_portfolio_sp_YTD['SPY YTD Close'] - 1

    # Cumulative sum of original investment
    merged_portfolio_sp_YTD['Cum Invst'] = merged_portfolio_sp_YTD['Cost Basis'].cumsum()

    # Cumulative sum of Ticker Share Value (latest FMV based on initial quantity purchased).
    merged_portfolio_sp_YTD['Cum Ticker Returns'] = merged_portfolio_sp_YTD['Ticker Share Value'].cumsum()

    # Cumulative sum of SP Share Value (latest FMV driven off of initial SP equiv purchase).
    merged_portfolio_sp_YTD['Cum SP Returns'] = merged_portfolio_sp_YTD['SPY Value'].cumsum()

    # Cumulative CoC multiple return for stock investments
    merged_portfolio_sp_YTD['Cum Ticker ROI Mult'] = merged_portfolio_sp_YTD['Cum Ticker Returns'] / merged_portfolio_sp_YTD['Cum Invst']

    merged_portfolio_sp_YTD['Current Value'] = merged_portfolio_sp_YTD['Last Close'] * merged_portfolio_sp_YTD['Quantity']
    merged_portfolio_sp_YTD['Pct Change'] = merged_portfolio_sp_YTD['Current Value'] / merged_portfolio_sp_YTD['Cost Basis']
    
    return merged_portfolio_sp_YTD

def portfolio_pnl(merged_portfolio_sp_YTD):
    merged_portfolio_sp_YTD.reset_index(inplace=True)
    df = merged_portfolio_sp_YTD
    df = df[['Ticker','Cost Basis','Current Value','Stock Gain / (Loss)','Pct Change']]
    return df

def calculate_daily_returns(df_closing_prices, portfolio_tickers):
    #Daily Returns
    df_daily_returns = pd.DataFrame()
    for ticker in portfolio_tickers:
        df_daily_returns[ticker] = df_closing_prices[ticker].pct_change()
    
    return df_daily_returns

def calculate_cumulative_daily_returns(df_daily_returns):
    # Calculate and plot the cumulative returns of the 4 fund portfolios 
    cum_daily_returns_df = (1 + df_daily_returns).cumprod() - 1
    cum_daily_returns_df.dropna(inplace=True)
    return cum_daily_returns_df

def calculate_std_deviation(df_daily_returns):
    df_daily_returns_dev= df_daily_returns.std()
    df_daily_returns_dev.sort_values(ascending=True)
    return df_daily_returns_dev

def calculate_annualized_std_dev(df_daily_returns_dev):
    # Calculate and sort the annualized standard deviation (252 trading days) of the portfolio
    # Review the annual standard deviations smallest to largest
    df_daily_returns_dev_252 = df_daily_returns_dev*np.sqrt(252)
    df_daily_returns_dev_252.sort_values(ascending=True)
    return df_daily_returns_dev_252

def calculate_rolling_21day_returns(df_daily_returns):
    df_daily_returns_std_21=df_daily_returns.rolling(window=21).std()
    df_daily_returns_std_21.dropna(inplace=True)
    return df_daily_returns_std_21

def calculate_annualized_returns(df_daily_returns):
    # Calculate the annualized Returns
    df_annual_returns=df_daily_returns.mean()*252
    df_annual_returns.sort_values(ascending=True)
    return df_annual_returns


def calculate_sharpe_ratio(df_annual_returns,df_daily_returns_dev_252):
    # Calculate the annualized Sharpe Ratios for each of the portfolio
    # Review the Sharpe ratios sorted lowest to highest
    df_sharpe=df_annual_returns/df_daily_returns_dev_252
    df_sharpe.sort_values(ascending=True)
    return df_sharpe


def calculate_SP500_daily_returns(df_closing_prices):
    df_spy_daily_returns = df_closing_prices['^GSPC'].pct_change()
    return df_spy_daily_returns

def calculate_rolling_60day_variance_SP500(df_closing_prices):
    # Calculate the variance of the SPY using a rolling 60-day window.
    df_spy_daily_returns = df_closing_prices['^GSPC'].pct_change()
    df_spy_var=df_spy_daily_returns.rolling(window=60).var()
    return df_spy_var

def compute_weights(portfolio_df,portfolio_tickers):
    # get the weights of the stocks in the portfolio using CostBasis
    weight_df = portfolio_df
    for ticker in portfolio_tickers:
        weight_df = portfolio_df['Cost Basis']/portfolio_df['Cost Basis'].sum()
        
    return weight_df
    
def calculate_rolling_60day_covariance(df_daily_returns,df_spy_daily_returns,portfolio_df,portfolio_tickers, weight_df):
    #Calculate Rolling Covariance 60 day
    rolling_covariance_60day = df_daily_returns.rolling(window=60).cov(df_spy_daily_returns)
    rolling_covariance_60day.tail()

    rolling_covariance_portfolio_60day = rolling_covariance_60day.dot(weight_df.to_list())

    return rolling_covariance_portfolio_60day

def calculate_rolling_portfolio_beta(rolling_covariance_portfolio_60day,df_spy_var):
    # Calculate the beta based on the 60-day rolling covariance compared to the market (S&P 500)
    # Review the last five rows of the beta information
    rolling_portfolio_beta=rolling_covariance_portfolio_60day/df_spy_var
    return rolling_portfolio_beta

def calculate_avg_rolling_portfolio_beta(rolling_portfolio_beta):
    # Calculate the average of the 60-day rolling beta
    avg_rolling_portfolio_beta = rolling_portfolio_beta.mean()
    return avg_rolling_portfolio_beta


def setup_monte_carlo_simulation(df_portfolio_ticker, weight_df, num_years=3,simulations=1000):
    # Set number of simulations
    num_sims = simulations

    # Configure a Monte Carlo simulation to forecast three years daily returns
    MC_Portfolio = MCSimulation(
        portfolio_data = df_portfolio_ticker,
        weights=weight_df.to_list(),
        num_simulation = num_sims,
        num_trading_days = 252*3
    )
    return MC_Portfolio
    
def run_monte_carlo_simulation(MC_Portfolio):
    MC_Portfolio.calc_cumulative_return()

def get_raw_portfolio_ticker_data(df_ticker):
    # Get the Portfolio without the S&P 500 numbers
    df_portfolio_ticker = df_ticker
    df_portfolio_ticker.columns = df_portfolio_ticker.columns.swaplevel(0, 1)
    df_portfolio_ticker.sort_index(axis=1, level=0, inplace=True)
    df_portfolio_ticker = df_portfolio_ticker.drop(columns='^GSPC')
    df_portfolio_ticker = df_portfolio_ticker.rename(columns={'Adj Close':'close'})
    return df_portfolio_ticker

def calculate_simulated_returns(MC_Portfolio):
    #Compute the Simulated Returns
    simulated_returns_data = {
        "mean": list(MC_Portfolio.simulated_return.mean(axis=1)),
        "median": list(MC_Portfolio.simulated_return.median(axis=1)),
        "min": list(MC_Portfolio.simulated_return.min(axis=1)),
        "max": list(MC_Portfolio.simulated_return.max(axis=1))
    }

    # Create a DataFrame with the summary statistics
    df_simulated_returns = pd.DataFrame(simulated_returns_data)

    # Display sample data
    return df_simulated_returns

def get_simulated_data(MC_Portfolio):
    #Display the Simulated Returns
    df_simulated_data=MC_Portfolio.simulated_return
    return df_simulated_data

def compute_cumulative_pnl(initial_investment, df_simulated_returns):
    
    # Multiply an initial investment by the daily returns of simulative stock prices to return 
    # the progression of daily returns in terms of money
    cumulative_pnl = initial_investment * df_simulated_returns

    # Display sample data
    return cumulative_pnl


def show_holdings(df):
    df_=df.copy()
    df_.index = [""] * len(df_)
    fmt = "%Y-%m-%d"
    styler = df_.style.format(
        {
            "Acquisition Date": lambda t: t.strftime(fmt),
            "Start of Year": lambda t: t.strftime(fmt),
            "Unit Cost": lambda t: "${:,.2f}".format(t),
            "Cost Basis": lambda t: "${:,.2f}".format(t)
        }
    )
    st.header("Holdings Summary")
    st.table(styler)
    #st.dataframe(df)
    gb = GridOptionsBuilder.from_dataframe(df_)

    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    #df2=df.copy()
    #df2.reset_index()
    #AgGrid(df_, gridOptions=gridOptions, enable_enterprise_modules=True)

def show_df(df,header):
    st.header(header)
    st.dataframe(df)
    
def show_table(df,header):
    st.header(header)
    fmt = "%Y-%m-%d"
    styler = df.style.format(
        {
            "Acquisition Date": lambda t: t.strftime(fmt),
            "Start of Year": lambda t: t.strftime(fmt),
            "Unit Cost": lambda t: "${:,.2f}".format(t),
            "Cost Basis": lambda t: "${:,.2f}".format(t),
            "Current Value": lambda t: "${:,.2f}".format(t),
            "Stock Gain / (Loss)": lambda t: "${:,.2f}".format(t),
            "Pct Change": lambda t: "{:,.2f}%".format(t*100.0)
        }
    )
    st.table(styler)
    df2=df.copy()
    df2.reset_index()
    gb = GridOptionsBuilder.from_dataframe(df2)

    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=False)
    gridOptions = gb.build()
    #AgGrid(df2, gridOptions=gridOptions, enable_enterprise_modules=True)

    
def show_header():
    st.title("Portfolio Analyzer")

    
def multi_plot(df, chartTitle, label, addAll = True):   
    fig = go.Figure(layout=go.Layout(height=600, width=1200))
    
    for column in df.columns.to_list():
        if column != 'Total':
            fig.add_trace(
                go.Scatter(
                    x = df.index,
                    y = df[column],
                    name = column
                 )
            )
        elif column == 'Total':
            fig.add_trace(
            go.Scatter(
                x = df.index,
                y = df[column],
                name = column,
                visible='legendonly'

                ) 
            )
    
    
    button_all = dict(label = 'All',
                      method = 'update',
                      args = [{'visible': df.columns.isin(df.columns),
                               'title': 'All',
                               'showlegend':True}])

    def create_layout_button(column):
        return dict(label = column,
                    method = 'update',
                    args = [{'visible': df.columns.isin([column]),
                             'title': column,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 1,
            buttons = ([button_all] * addAll) + 
                list(df.columns.map(lambda column: create_layout_button(column)))
            
            )
            
            
        ],title_text=chartTitle)
    return fig


def create_and_show_charts(merged_portfolio_sp_YTD,df_portfolio_value
                           ,df_daily_returns
                          ,cum_daily_returns_df
                          ,df_daily_returns_std_21
                          ,df_sharpe
                          ,rolling_portfolio_beta 
                          ,df_simulated_data
                          ,cumulative_pnl
                          ,simulation_summary
                          ,beta_summary
                          ):
    df_portfolio_pnl = portfolio_pnl(merged_portfolio_sp_YTD)
    df_portfolio_pnl.set_index(['Ticker'], inplace=True)

    
    #Sum up by Row totals
    df_portfolio_value.loc[:,'Total'] = df_portfolio_value.sum(axis=1)
    # this ends up skewing the chart

    fig_current_pnlvalue = px.pie(merged_portfolio_sp_YTD, 
                                  values='Stock Gain / (Loss)', names='Ticker', 
                                  title='<b>Portfolio Current P&L</b>',height=400, width=800
                          )
    fig_current_value = px.pie(merged_portfolio_sp_YTD, 
                               values='Current Value', names='Ticker', 
                               title='<b>Portfolio Current Value</b>',height=400, width=800
                          )
    
    # YTD Charts

    trace1 = go.Bar(
        x = merged_portfolio_sp_YTD['Ticker'][0:10],
        y = merged_portfolio_sp_YTD['Share YTD'][0:10],
        name = 'Ticker YTD')

    trace2 = go.Scatter(
        x = merged_portfolio_sp_YTD['Ticker'][0:10],
        y = merged_portfolio_sp_YTD['SPY YTD'][0:10],
        name = 'SPY YTD')

    data1 = [trace1, trace2]

    layout1 = go.Layout(title = '<b>YTD Return vs SPY YTD</b>'
        , barmode = 'group'
        , yaxis=dict(title='Returns', tickformat=".2%")
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.8,y=1)
        ,height=800, width=1600
        )

    fig1 = go.Figure(data=data1, layout=layout1)
    
    #Total return Comparison Charts
    trace3 = go.Bar(
    x = merged_portfolio_sp_YTD['Ticker'][0:10],
    y = merged_portfolio_sp_YTD['Ticker Return'][0:10],
    name = 'Ticker Total Return')

    trace4 = go.Scatter(
        x = merged_portfolio_sp_YTD['Ticker'][0:10],
        y = merged_portfolio_sp_YTD['SP Return'][0:10],
        name = 'SP500 Total Return')

    data2 = [trace3, trace4]

    layout2 = go.Layout(title = '<b>Total Return vs S&P 500</b>'
        , barmode = 'group'
        , yaxis=dict(title='Returns', tickformat=".2%")
        , xaxis=dict(title='Ticker', tickformat=".2%")
        , legend=dict(x=.8,y=1)
        ,height=800, width=1600
        )

    fig2 = go.Figure(data=data2, layout=layout2)
    
    #Gain Loss Total Return vs S&P 500
    trace5 = go.Bar(
    x = merged_portfolio_sp_YTD['Ticker'][0:10],
    y = merged_portfolio_sp_YTD['Stock Gain / (Loss)'][0:10],
    name = 'Ticker Total Return ($)')

    trace6 = go.Bar(
        x = merged_portfolio_sp_YTD['Ticker'][0:10],
        y = merged_portfolio_sp_YTD['SPY Gain / (Loss)'][0:10],
        name = 'SPY Total Return ($)')

    trace7 = go.Scatter(
        x = merged_portfolio_sp_YTD['Ticker'][0:10],
        y = merged_portfolio_sp_YTD['Ticker Return'][0:10],
        name = 'Ticker Total Return %',
        yaxis='y2')

    data3 = [trace5, trace6, trace7]

    layout3 = go.Layout(title = '<b>Gain / (Loss) Total Return vs SPY</b>'
        , barmode = 'group'
        , yaxis=dict(title='Gain / (Loss) ($)')
        , yaxis2=dict(title='Ticker Return', overlaying='y', side='right', tickformat=".2%")
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.75,y=1)
        ,height=800, width=1600
        )

    fig3 = go.Figure(data=data3, layout=layout3)
    
    #Total Cumulative Investments
    trace8 = go.Bar(
    x = merged_portfolio_sp_YTD['Ticker'],
    y = merged_portfolio_sp_YTD['Cum Invst'],
    # mode = 'lines+markers',
    name = 'Cum Invst')

    trace9 = go.Bar(
        x = merged_portfolio_sp_YTD['Ticker'],
        y = merged_portfolio_sp_YTD['Cum SP Returns'],
        # mode = 'lines+markers',
        name = 'Cum SP500 Returns')

    trace10 = go.Bar(
        x = merged_portfolio_sp_YTD['Ticker'],
        y = merged_portfolio_sp_YTD['Cum Ticker Returns'],
        # mode = 'lines+markers',
        name = 'Cum Ticker Returns')

    trace11 = go.Scatter(
        x = merged_portfolio_sp_YTD['Ticker'],
        y = merged_portfolio_sp_YTD['Cum Ticker ROI Mult'],
        # mode = 'lines+markers',
        name = 'Cum ROI Mult'
        , yaxis='y2')


    data4 = [trace8, trace9, trace10, trace11]

    layout4 = go.Layout(title = '<b>Total Cumulative Investments Over Time</b>'
        , barmode = 'group'
        , yaxis=dict(title='Returns')
        , xaxis=dict(title='Ticker')
        , legend=dict(x=.4,y=1)
        , yaxis2=dict(title='Cum ROI Mult', overlaying='y', side='right')  
        ,height=800, width=1600
        )

    fig4 = go.Figure(data=data4, layout=layout4)
    
    fig_daily_returns = multi_plot(df_daily_returns,'<b>Daily Returns</b>',"Daily Returns")
    
    fig_cum_daily_returns = px.box(cum_daily_returns_df, title = '<b>Cumulative  Returns</b>'
              ,height=800, width=1600
        )
    
    fig_rolling_21day_returns = multi_plot(df_daily_returns_std_21, '<b>Rolling 21 Day Returns</b>',"21 Day Returns")
    
    fig_sharpe_ratio= px.bar(df_sharpe,title='<b>Sharpe Ratio</b>',
                      labels={"value": "Sharpe Ratio"}
                      #,template='simple_white'
                      ,height=800, width=1600
                     )
    fig_rolling_60day_beta = px.line(rolling_portfolio_beta,title='<b>Rolling 60 Day Beta of the Portfolio</b>',
                      labels={"value": "Beta"}
                      #,template='simple_white'
                      ,height=800, width=1600
                     )
    
    fig_simulated_returns = px.line(df_simulated_data, 
                          title='<b>Simulated Returns of the Portfolio</b>',
                          labels={"value": "Returns"}
                          ,height=800, width=1600
                         )
    
    fig_simulated_cum_pnl = px.line(cumulative_pnl, 
                          title='<b>Simulated Cumulative P&L<b>',
                          labels={"value": "Returns"}
                          ,height=800, width=1600
                         )
    
    #############################################################
    # Display all the results
    #############################################################
    
    show_table(df_portfolio_pnl,"Portfolio PNL")
    #st.plotly_chart(fig_current_pnlvalue,use_container_width=True)
    #st.plotly_chart(fig_current_value,use_container_width=True)
    
    col1,col2 = st.columns(2) 
    with col1:
        st.plotly_chart(fig_current_pnlvalue,use_container_width=True)
    with col2:
        st.plotly_chart(fig_current_value,use_container_width=True)

    fig_portfolio_value = multi_plot(df_portfolio_value,'<b>Portfolio Value<b>', 'Portfolio Value')
    st.plotly_chart(fig_portfolio_value,use_container_width=True)
    
    st.plotly_chart(fig_daily_returns, use_container_width=True)
    st.plotly_chart(fig_cum_daily_returns, use_container_width=True)
    st.plotly_chart(fig_rolling_21day_returns, use_container_width=True)
    st.plotly_chart(fig_sharpe_ratio, use_container_width=True)
    st.plotly_chart(fig_rolling_60day_beta, use_container_width=True)
    st.text(beta_summary)
    
    st.plotly_chart(fig_simulated_returns, use_container_width=True)
    st.plotly_chart(fig_simulated_cum_pnl, use_container_width=True)
    st.text(simulation_summary)
    
    st.plotly_chart(fig1,use_container_width=True)
    st.plotly_chart(fig2,use_container_width=True)
    st.plotly_chart(fig3,use_container_width=True)
    st.plotly_chart(fig4,use_container_width=True)


def run_analysis(portfolio_df):
    
    #Get Dates and Tickers
    portfolio_tickers = get_portfolio_tickers(portfolio_df)
    tickers = get_tickers(portfolio_df)
    end_last_year = get_YTD_date(portfolio_df)
    
    #Get Real-time data
    df_ticker = get_ticker_data(portfolio_df, tickers)
    df_closing_prices=get_closing_prices(df_ticker,tickers)
    
    #Get the latest closes
    df_close_latest=get_latest_closes(df_closing_prices)
    #show_df(df_close_latest,"Latest Closes")
    
    df_close_YTD_start = get_YTD_closes(df_closing_prices,end_last_year)
    #show_df(df_close_YTD_start,"YTD Start Closes")
    
    #Switch the portfolio dataframe to using the Ticker as an index
    portfolio_df.set_index(['Ticker'], inplace=True)
    df_portfolio_value = get_portfolio_closes(portfolio_df,df_closing_prices,portfolio_tickers)
    
    #Get the latest Closes, Merge with Portfolio and compute Numbers
    merged_portfolio = merge_portfolio_with_closes(portfolio_df, df_close_latest)
    merged_portfolio = calculations_with_last_close(merged_portfolio)
    
    #show_table(merged_portfolio,"Portfolio with closes")
    
    #Find SPY Closes, merge to Portfolio by Acquisition Date, and compute SPY Equivalent Trades
    df_spy_closing_prices = get_SPY_closing_prices(df_closing_prices)
    merged_portfolio_sp = merge_portfolio_with_SPY(merged_portfolio, df_spy_closing_prices)
    merged_portfolio_sp = calculations_with_SPY(merged_portfolio_sp,df_closing_prices)
    
    #show_table(merged_portfolio_sp,"Portfolio with SPY Closes")
    
    #Get YTD Start Closes and compute YTD Stats
    merged_portfolio_sp_YTD = merge_portfolio_with_YTD(merged_portfolio_sp,df_close_YTD_start)
    merged_portfolio_sp_YTD = calculate_YTD(merged_portfolio_sp_YTD, df_closing_prices, end_last_year)
    
    #show_table(merged_portfolio_sp_YTD,"Portfolio with YTD")
    
    weight_df = compute_weights(portfolio_df,portfolio_tickers)
    
    df_daily_returns = calculate_daily_returns(df_closing_prices, portfolio_tickers)
    
    cum_daily_returns_df = calculate_cumulative_daily_returns(df_daily_returns)
    
    df_daily_returns_dev = calculate_std_deviation(df_daily_returns)
    
    
    df_daily_returns_dev_252 = calculate_annualized_std_dev(df_daily_returns_dev)
    
    df_daily_returns_std_21 = calculate_rolling_21day_returns(df_daily_returns)
    
    df_annual_returns = calculate_annualized_returns(df_daily_returns)
    
    df_sharpe = calculate_sharpe_ratio(df_annual_returns,df_daily_returns_dev_252)
    
    df_spy_var = calculate_rolling_60day_variance_SP500(df_closing_prices)
    df_spy_daily_returns = calculate_SP500_daily_returns(df_closing_prices)
    
    rolling_covariance_portfolio_60day = calculate_rolling_60day_covariance(
        df_daily_returns,df_spy_daily_returns,portfolio_df,portfolio_tickers, weight_df)
    
    
    rolling_portfolio_beta = calculate_rolling_portfolio_beta(rolling_covariance_portfolio_60day,df_spy_var)
    
    
    avg_rolling_portfolio_beta = calculate_avg_rolling_portfolio_beta(rolling_portfolio_beta)
    
    beta_summary = f"Average 60 Day Rolling Portfolio Beta is {avg_rolling_portfolio_beta:.2f}"

    df_portfolio_ticker = get_raw_portfolio_ticker_data(df_ticker)
    
    MC_Portfolio = setup_monte_carlo_simulation(df_portfolio_ticker, weight_df,3,300)
    run_monte_carlo_simulation(MC_Portfolio)
    
    df_simulated_returns = calculate_simulated_returns(MC_Portfolio)
    

    df_simulated_data = get_simulated_data(MC_Portfolio)
    
    # Set initial investment to the Current Value for cumulative_pnl calc
    initial_investment = merged_portfolio_sp_YTD['Current Value'].sum()
    cumulative_pnl = compute_cumulative_pnl(merged_portfolio_sp_YTD['Current Value'].sum(), df_simulated_returns) 
        
    
    
    # Fetch summary statistics from the Monte Carlo simulation results
    tbl = MC_Portfolio.summarize_cumulative_return()
    # Compute the upper bound and lower bound with 95% confidence
    ci_lower = round(tbl[8]*initial_investment,2)
    ci_upper = round(tbl[9]*initial_investment,2)
    simulation_summary =f"There is a 95% chance that a current investment of ${initial_investment:,.2f} in the portfolio over the next year will end within in the range of ${ci_lower:,.2f} and ${ci_upper:,.2f}."

    create_and_show_charts(merged_portfolio_sp_YTD,df_portfolio_value, df_daily_returns
                          ,cum_daily_returns_df
                          ,df_daily_returns_std_21
                          ,df_sharpe
                          ,rolling_portfolio_beta
                          ,df_simulated_data
                          ,cumulative_pnl 
                          ,simulation_summary
                          ,beta_summary
                          )
            
def main():
    st.set_page_config(layout='wide')
    show_header()
    holdings_df = get_holdings()
    show_holdings(holdings_df)
    if st.button('Start Analysis'):
        with st.spinner("Analysis Running"):
            #Refresh the Portfolio before every run
            df_portfolio = pd.DataFrame()
            df_portfolio = get_holdings()
            df_portfolio.reset_index(inplace=True)
            run_analysis(df_portfolio)
    

if __name__ == "__main__":
    main()
