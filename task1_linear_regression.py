import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Load training data
df = pd.read_csv("train.csv")

# Features to use
features = ['GrLivArea', 'FullBath', 'BedroomAbvGr']
X = df[features]
y = df['SalePrice']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on validation set
y_pred = model.predict(X_test)

# 5. Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
print("Model Coefficients:", dict(zip(features, model.coef_)))
print("Model Intercept:", model.intercept_)

# 6. Plot Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linewidth=2)
plt.show()


# --------------------------------------------------------
# ⭐ ADDED GRAPH 2: Square Footage vs Price + Regression Line
# --------------------------------------------------------

plt.figure(figsize=(8, 5))

# scatter of actual points
plt.scatter(df["GrLivArea"], df["SalePrice"],
            color="skyblue", alpha=0.5, label="Actual Data")

# regression line based only on GrLivArea
x_line = np.linspace(df["GrLivArea"].min(), df["GrLivArea"].max(), 200)
y_line = (model.coef_[0] * x_line) + model.intercept_

plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression Line")

plt.title("Linear Regression Trend: Living Area (sq ft) vs Sale Price")
plt.xlabel("GrLivArea (Square Footage)")
plt.ylabel("SalePrice (Price)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7. Predict on test.csv
test_df = pd.read_csv("test.csv")
X_test_final = test_df[features].fillna(0)  # fill missing values with 0
test_pred = model.predict(X_test_final)

# 8. Create submission.csv
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")