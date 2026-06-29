
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)

df = pd.concat(
    [housing.data, housing.target.rename("HousePrice")],
    axis=1
)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.hist(figsize=(15,10), bins=30)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print(results_df)

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="Model", y="R2 Score")
plt.title("Model Comparison (R² Score)")
plt.xticks(rotation=10)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(data=results_df, x="Model", y="RMSE")
plt.title("Model Comparison (RMSE)")
plt.xticks(rotation=10)
plt.show()

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

if best_model_name == "Decision Tree":
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))
    sns.barplot(data=importance, x="Importance", y="Feature")
    plt.title("Feature Importance")
    plt.show()

    print(importance)

joblib.dump(best_model, "best_house_price_model.pkl")

loaded_model = joblib.load("best_house_price_model.pkl")

sample_house = [[
    8.3252,
    41,
    6.984127,
    1.023810,
    322,
    2.555556,
    37.88,
    -122.23
]]

sample_house_scaled = scaler.transform(sample_house)

prediction = loaded_model.predict(sample_house_scaled)

print("Best Model:", best_model_name)
print("Predicted House Price:", prediction[0])
