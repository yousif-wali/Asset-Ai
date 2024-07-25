import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from technical_analysis import Technical
from sklearn.feature_selection import RFE

# Load the data
data = pd.read_csv("./spy_data.csv")

# Filter data for AAPL ticker
data = data[data["Ticker"] == "AAPL"]

# Calculate technical indicators
ta = Technical(data)
print("Loading Indicators.")
data["RSI_14"] = ta.calculate_rsi(14)
data["MA_100"] = ta.calculate_ma(100)
data["ADX_14"], data["ATR_14"] = ta.calculate_adx(14)
data["MACD"], data["Signal"], data["MACD_Hist"] = ta.calculate_macd()
data["Momentum"] = ta.calculate_momentum(10)
data["Squeeze_Momentum"], data["Squeeze_Momentum_Indicator"] = ta.calculate_squeeze_momentum()
data["%K"], data["%D"] = ta.calculate_stochastic()
data["Wyckoff_Accumulation"] = ta.detect_wyckoff_accumulation()
data["Wyckoff_Distribution"] = ta.detect_wyckoff_distribution()
data["Double_Top"] = ta.detect_double_top()
data["Double_Bottom"] = ta.detect_double_bottom()
print("All Indicators are ready.")

# Drop rows with NaN values
data = data.dropna()

# Select features and target
selected_features = ["MA_100", "ADX_14", "MACD", "MACD_Hist", "Wyckoff_Distribution", "Double_Top", "Double_Bottom"]
X = data[selected_features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Our target is the close price
y = data["Close"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature selection using Recursive Feature Elimination (RFE)
model = LinearRegression()
rfe = RFE(model, n_features_to_select=len(selected_features))
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train the Linear Regression model
model.fit(X_train_rfe, y_train)

# Make predictions
y_pred = model.predict(X_test_rfe)

# Evaluate the model
mse = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')

# Store results
results = {
    "Linear Regression": {
        "Mean Absolute Error": mse,
        "R2 Score": r2,
        "Cross-Validation R2 Scores": cv_scores,
        "Mean Cross-Validation R2 Score": cv_scores.mean()
    }
}

# Print results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"  Mean Absolute Error: {result['Mean Absolute Error']}")
    print(f"  R2 Score: {result['R2 Score']}")
    print(f"  Cross-Validation R2 Scores: {result['Cross-Validation R2 Scores']}")
    print(f"  Mean Cross-Validation R2 Score: {result['Mean Cross-Validation R2 Score']}")
    print()

def heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_scaled, columns=selected_features).corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Uncomment to show heatmap
#heatmap()
