
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

## Run Configurations
### Dash Configurations
Run PortfolioAnalysis.ipynb file and view the results in http://localhost:8096

### Streamlit local
Run the command: streamlit run streamlit_app.py
and view the results in http://localhost:8501

### Streamlit cloud
View the results in https://share.streamlit.io/sumeetvaidya/bootcamp-project-1/main

## Dashboard Images

<img width="1423" alt="Screen Shot 2021-11-13 at 9 53 35 PM" src="https://user-images.githubusercontent.com/17937188/141665624-2c01317d-c758-4a2d-b7e0-e921b36a679f.png">
<img width="1431" alt="Screen Shot 2021-11-13 at 9 54 13 PM" src="https://user-images.githubusercontent.com/17937188/141665637-a83d13dd-4082-405e-a7f0-4416c958ec3d.png">

- - - <img width="1370" alt="Screen Shot 2021-11-13 at 9 59 34 PM" src="https://user-images.githubusercontent.com/17937188/141665750-fe924c2f-4bba-48e1-9cf9-2d66c27e162a.png">

<img width="1323" alt="Screen Shot 2021-11-13 at 10 01 02 PM" src="https://user-images.githubusercontent.com/17937188/141665772-74fe1f83-1c54-4646-ae8c-ca6f9b027021.png">
<img width="1341" alt="Screen Shot 2021-11-13 at 10 04 04 PM" src="https://user-images.githubusercontent.com/17937188/141665827-c7f9a79c-204f-4227-8bbf-7b235a10e409.png">

<img width="1301" alt="Screen Shot 2021-11-13 at 10 04 28 PM" src="https://user-images.githubusercontent.com/17937188/141665832-86dc4cef-ef01-4413-866a-ecb6b92687c7.png">
<img width="1279" alt="Screen Shot 2021-11-13 at 10 04 49 PM" src="https://user-images.githubusercontent.com/17937188/141665836-54d5ae1f-a107-4e60-9cf4-91bebf8032da.png">
<img width="1295" alt="Screen Shot 2021-11-13 at 10 05 15 PM" src="https://user-images.githubusercontent.com/17937188/141665847-35f9c4bb-5fa9-4247-9b8e-4f70f387598f.png">

<img width="1297" alt="Screen Shot 2021-11-13 at 10 06 39 PM" src="https://user-images.githubusercontent.com/17937188/141665883-c2a1aa5e-91eb-4e5b-8cfc-428f4c03fd25.png">
<img width="1379" alt="Screen Shot 2021-11-13 at 10 08 50 PM" src="https://user-images.githubusercontent.com/17937188/141665914-71fb5bcc-6014-4060-86ad-017109dca420.png">


<img width="1318" alt="Screen Shot 2021-11-13 at 10 10 06 PM" src="https://user-images.githubusercontent.com/17937188/141665937-76821dc2-760b-41fa-9fd7-0cf698578e15.png">

<img width="1336" alt="Screen Shot 2021-11-13 at 10 11 04 PM" src="https://user-images.githubusercontent.com/17937188/141665954-f6af74a4-8d21-4e1c-827d-caf3d7c83039.png">

<img width="1240" alt="Screen Shot 2021-11-13 at 10 11 30 PM" src="https://user-images.githubusercontent.com/17937188/141665963-24acdcdf-acd0-42e4-a4db-d9f792451cd1.png">

<img width="1246" alt="Screen Shot 2021-11-13 at 10 13 12 PM" src="https://user-images.githubusercontent.com/17937188/141665991-9839a9ff-9725-4d5d-abcd-a0dbf92aad27.png">

<img width="1201" alt="Screen Shot 2021-11-13 at 10 13 46 PM" src="https://user-images.githubusercontent.com/17937188/141666012-38208a18-313d-49bb-8621-a4647f18d96f.png">
<img width="1296" alt="Screen Shot 2021-11-13 at 10 14 35 PM" src="https://user-images.githubusercontent.com/17937188/141666033-91e33997-5559-4504-a243-17d57f55dbab.png">

<img width="1251" alt="Screen Shot 2021-11-13 at 10 15 04 PM" src="https://user-images.githubusercontent.com/17937188/141666042-6330d5a3-2671-4c4d-a0c8-4e4c2d0a5e0b.png">

<img width="1282" alt="Screen Shot 2021-11-13 at 10 15 49 PM" src="https://user-images.githubusercontent.com/17937188/141666055-bb06aade-4b58-44ed-b906-d7d436e54c01.png">
<img width="1330" alt="Screen Shot 2021-11-13 at 10 16 26 PM" src="https://user-images.githubusercontent.com/17937188/141666067-ed4447e0-c7f7-4af2-9162-01291263be7a.png">

<img width="1292" alt="Screen Shot 2021-11-13 at 10 16 52 PM" src="https://user-images.githubusercontent.com/17937188/141666075-f767259d-3c3c-47f9-a431-25a2612aab48.png">


## Technologies
This is a Python v 3.7 project leveraging numerous python modules. The modules are to be imported to the main project file.

####  Modules

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

####  APIs and Datasources
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

