from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  

file_path = "pip_dataset_selected.csv"
df_final = pd.read_csv(file_path)

 
df_numeric = df_final.drop(columns=["country", "welfare_type"])

df_ind = df_numeric.fillna(df_numeric.median(numeric_only=True))

 
X = df_ind.drop(columns=["headcount_ratio_international_povline"])  
y = df_ind["headcount_ratio_international_povline"]  

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

 
y_pred = linear_regressor.predict(X_test)

 
mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred)

cross_val = cross_val_score(linear_regressor, X, y, cv=5, scoring='r2')


linear_regression_results = {
    "Mean Absolute Error": mae_lr,
    "Mean Squared Error": mse_lr,
    "Root Mean Squared Error": rmse_lr,
    "R2 Score": r2_lr,
    "Cross-Validation R2 Mean": np.mean(cross_val),
    "Cross-Validation R2 Std": np.std(cross_val)
}

print("\n Linear Regression Model Performance ")
for metric, value in linear_regression_results.items():
    print(f"{metric}: {value:}")

# Visualization: Actual vs. Predicted Poverty Rates
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")  
plt.xlabel("Actual Poverty Rate")
plt.ylabel("Predicted Poverty Rate")
plt.title("Linear Regression: Actual vs. Predicted Poverty Rates")
plt.show()