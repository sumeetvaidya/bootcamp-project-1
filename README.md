
## Portfolio Analyzer
This project uses yFinance, Python, and the Dash library to analyze and display the performance of a stock portfolio in a Dashboard. The following calculations are then displayed:

* analyzes the current P&L
* compares with S&P 500
* YTD Return vs SPY YTD
* Total Return vs SP500
* P&L Total Return SPY
* Total Cumulative Return over time
* Daily Returns, 
* Cumulative Return by Ticker
* Rolling 21 day return
* Sharpe Ratio
* Rolling 60 day Beta
* Simulated Returns
* Simulated Cumulative P&L

The results are displayed in Dash/Plotly and in Streamlit. The application is also available in Streamlit in the cloud via https://share.streamlit.io/sumeetvaidya/bootcamp-project-1/main

This project was built by Sumeet Vaidya, Pat Beeson, William Alford and Scott Oziros

## Dashboard Images
![alt text]()
![alt text]()

- - - 
## Technologies

This is a Python v 3.7 project leveraging numerous python modules. The modules are to be imported to the main project file.

#### -- Modules

This project  uses Jupyter Notebook while the presentation layer uses Dash for the interactive visual dashboard.
The Alternate implentation uses streamlit.
 * pandas
 * numpy
 * matplotlib
 * dash
 * plotly
 * yfinance
 * MCForecastTools
 * streamlit

#### -- APIs and Datasources
The project also leverages three years of daily stocks trades from Yahoo! Finanace for the symbols in the Portfolio and the S&P 500 Index.

- - - 
## Installation Guide
The project requires the following environments to be installed in the main project file via a command line system:

```
conda create -n project python=3.7 anaconda -y
conda activate project
conda install -c pyviz hvplot geoviews
pip install python-dotenv
pip install alpaca-trade-api
conda install -c anaconda requests
conda install ipykernel
conda install nb_conda_kernels
conda install dash
conda install -c plotly jupyter-dash
conda install -c plotly jupyterlab-dash
pip install streamlit
pip install streamlit-aggrid
conda install pandas
```
## Module version Numbers
* pandas==1.3.4
* pandas-datareader==0.10.0
* numpy==1.20.3
* plotly==4.14.3
* streamlit==1.2.0
* streamlit-aggrid==0.2.2.post4
* yfinance==0.1.64
* alpaca-trade-api==1.4.1
* openpyxl==3.0.9




As a resource, the following link is to the Python 3.7 Reference Guide 

[Python documentation](https://docs.python.org/3.7/)


- - - 
## Contributors
This project was built by Sumeet Vaidya, Pat Beeson, William Alford and Scott Oziros
This is a group student project for Columbia University FinTech bootcamp.

- - - 
## License
Any usage of this project should be authorized from Columbia Univesity bookcamp.

