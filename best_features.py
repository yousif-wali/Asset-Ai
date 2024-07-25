import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from technical_analysis import Technical

pd.options.mode.chained_assignment = None

# Load the data
data = pd.read_csv("./spy_data.csv")

# Filter data for AAPL ticker
data = data[data["Ticker"] == "MSFT"]

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

# List of all features
all_features = ["Volume", "MA_100", "RSI_14", "ADX_14", "MACD", "MACD_Hist", "Squeeze_Momentum", "Wyckoff_Accumulation", "Wyckoff_Distribution", "Double_Top", "Double_Bottom"]

# Standardize the entire feature set once
scaler = StandardScaler()
data[all_features] = scaler.fit_transform(data[all_features])

# Target variable
y = data["Close"]

# Function to evaluate a set of features
def evaluate_features(features):
    X = data[features]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

    return {
        "features": features,
        "Mean Absolute Error": mse,
        "R2 Score": r2,
        "Cross-Validation R2 Scores": cv_scores,
        "Mean Cross-Validation R2 Score": cv_scores.mean()
    }

# Generate different sets of features and evaluate the model for each set
results = []
for r in range(1, len(all_features) + 1):
    for subset in combinations(all_features, r):
        result = evaluate_features(list(subset))
        results.append(result)

# Find the best set of features based on Mean Cross-Validation R2 Score
best_result = max(results, key=lambda x: x["Mean Cross-Validation R2 Score"])

# Print the best result
print("Best Feature Set:")
print(f"Features: {best_result['features']}")
print(f"Mean Absolute Error: {best_result['Mean Absolute Error']}")
print(f"R2 Score: {best_result['R2 Score']}")
print(f"Cross-Validation R2 Scores: {best_result['Cross-Validation R2 Scores']}")
print(f"Mean Cross-Validation R2 Score: {best_result['Mean Cross-Validation R2 Score']}")

# Optional: Plot heatmap of the best feature set
def heatmap(features):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[features].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

heatmap(best_result['features'])