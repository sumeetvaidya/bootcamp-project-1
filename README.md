# Project 1
## Portfolio Analyzer
This project ....

The project is deployed as a ...

## Images of the Dashboard 
![alt text]()
![alt text]()

- - - 
## Technologies

This is a Python v 3.7 project leveraging numerous python modules. The modules are to be imported to the main project file.



####  Modules
Modules to be imported into an editor. This project primarily uses Jupyter Notebook while the presentation layer uses Dash for the interactive visual dashboard.
```
import pandas as pd
import numpy as np
from pathlib import Path
import sqlalchemy
%matplotlib inline
import hvplot.pandas
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
from dash import html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from MCForecastTools import MCSimulation
import dash_table
```

#### APIs and Datasources
Alpaca SDK was used to create the Alpaca tradeapi.REST object to fecth source data. Reqister for an account with Alpaca to receive a Secret Key and API Key. Create and save an environment file locally in a seperate file from the main project file.

Click this [link](https://app.alpaca.markets/login) to register and receive your keys.

####  Database Connection String
Database connection string and the physical database used queries and combining multiple historical daily trades source data.

database_connection_string ='sqlite:///Muskies.db'

The project also leverages ten years of daily stocks trades from Yahoo Finanace for the following Indices: Crypto, Bonds, SP500 and the Gold Index.

- - - 
#### Installation Guide
The project requires the following environments to be installed via a command line system:

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
```

Python 3.7 Reference Guide

[Python documentation](https://docs.python.org/3.7/)


- - - 
#### Contributors
This is a group project for Columbia University FinTech bootcamp.

- - - 
#### License
Any usage of this project should be authorized from Columbia Univesity bookcamp.

