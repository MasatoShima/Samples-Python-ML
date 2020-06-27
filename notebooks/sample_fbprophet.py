# %%
"""
Name: sample_fbprophet.py
Created by: Masato Shima
Created on: 2020/06/24
Description:
	Hands-on
	Time-Series Forecasting: Predicting Stock Prices Using Facebook’s Prophet Model
	https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-facebooks-prophet-model-9ee1657132b5
"""

# %%
# **************************************************
# ----- Import Library
# **************************************************
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet


# %%
# **************************************************
# ----- Load & inspect the data
# **************************************************
data = pd.read_csv(
	"./data/price_GOOG.csv",
	header=0,
	encoding="utf-8"
)

data.head(5)

data.describe()


# %%
# **************************************************
# ----- Build the predictive model
# **************************************************
# Select only the important features i.e. the date and price
# Select Date and Price
data = data[["Date", "Close"]]

# Rename the features: These names are NEEDED for the model fitting
# Renaming the columns of the dataset
data = data.rename(
	columns={
		"Date": "ds",
		"Close": "y"
	}
)

data.head(5)


# %%
# The Prophet class (model)
m = Prophet(daily_seasonality=True)

# Fit the model using all data
m.fit(data)


# %%
# **************************************************
# ----- Plot the predictions
# **************************************************
# We need to specify the number of days in future
future = m.make_future_dataframe(periods=365)
prediction = m.predict(future)

m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()


# %%
# **************************************************
# ----- Plot the trend, weekly, seasonally, yearly and daily components
# **************************************************
m.plot_components(prediction)
plt.show()


# **************************************************
# ----- End
# **************************************************
