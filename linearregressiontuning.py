import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("pip_dataset_selected.csv")
df_numeric = df.drop(columns=["country", "welfare_type"])
df_cleaned = df_numeric.fillna(df_numeric.median(numeric_only=True))


X = df_cleaned.drop(columns=["headcount_ratio_international_povline"])
y = df_cleaned["headcount_ratio_international_povline"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


alpha_values = {'alpha': [0.01, 0.1, 1, 10, 100]}
 
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)

 
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, alpha_values, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

 
lasso = Lasso(max_iter=10000)
lasso_grid = GridSearchCV(lasso, alpha_values, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso = lasso_grid.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)


# Linear Regression Metrics
r2_linear = r2_score(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

# Ridge Regression Metrics
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

# Lasso Regression Metrics
r2_lasso = r2_score(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
 
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'R² Score': [r2_linear, r2_ridge, r2_lasso],
    'MAE': [mae_linear, mae_ridge, mae_lasso],
    'RMSE': [rmse_linear, rmse_ridge, rmse_lasso]
})

print("\nModel Performance Comparison:")
print(results)


# Ridge visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_ridge, color='blue', alpha=0.6, label='Predicted (Ridge)')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Ridge Regression: Actual vs. Predicted')
plt.xlabel('Actual Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lasso visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lasso, color='green', alpha=0.6, label='Predicted (Lasso)')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Lasso Regression: Actual vs. Predicted')
plt.xlabel('Actual Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Linear Regression visualization
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, color='purple', alpha=0.6, label='Predicted (Linear)')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal Line')
plt.title('Linear Regression: Actual vs. Predicted')
plt.xlabel('Actual Poverty Rate')
plt.ylabel('Predicted Poverty Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()