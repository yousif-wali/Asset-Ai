# Crypto-Ai
This Supervised Machine Learning can forecast any asset's price movements with high R2 score. It uses Linear Regression model, also theres other models in the code too. Please read the readme file


To download the data used in this tutorial.
https://www.kaggle.com/datasets/joebeachcapital/s-and-p500-index-stocks-daily-updated


We used S&P500 index, which includes about 500 different stock assets. You can try each subset of data by filtering them
<code>
  data = data[data["Ticker"] == "APA"]
</code>

<b>Please first take a look at <i><u>best_features.py</u></i> file, as it is focusing on getting the best feature into the model.</b>

After the prefferabled feature/s of the asset has been chosen, then you can alter the <code>selected_features</code> in <i><u>index.py</u></i> file


<b>Note: if you try to use a different dataset make sure it include the following columns (High, Low, Close, Volume) as they need to be used in <i>technical_analysis.py</i> class</b>

if your dataset is designated to only one asset, then you can remove
<code>
  data = data[data["Ticker"] == "APA"]
</code>
as it is only to filter out rows/data for only specified stock/asset
