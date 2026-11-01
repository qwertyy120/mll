import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    'Rooms': [2, 3, 4, 5, 6, 3, 4, 5, 6, 7],
    'Area': [500, 600, 700, 800, 900, 600, 750, 850, 950, 1000],
    'Price': [20, 30, 40, 50, 60, 32, 42, 52, 62, 70]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[['Rooms', 'Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMAE:", mean_absolute_error(y_test, y_pred))

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

print("RMSE:", np.sqrt(mse))

print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot([minimum, maximum], [minimum, maximum], 'r--')

plt.show()
