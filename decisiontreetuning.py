import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("pip_dataset_selected.csv")
df_numeric = df.drop(columns=["country", "welfare_type"])
df_cleaned = df_numeric.fillna(df_numeric.median(numeric_only=True))

# Define features and labels
X = df_cleaned.drop(columns=["headcount_ratio_international_povline"])
y = df_cleaned["headcount_ratio_international_povline"]

# Convert poverty rate into categories
y_class = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criteria': ['gini', 'entropy']
}

# Initialize classifier and grid search
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best model
best_tree = grid_search.best_estimator_

# Evaluate performance
y_pred = best_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(best_tree, X, y_class, cv=5, scoring='accuracy')

# Output results
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy)
print("Cross-Validation Accuracy Mean:", np.mean(cv_scores))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(16, 8))
plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=["Low", "Medium", "High"])
plt.title("Tuned Decision Tree Classifier")
plt.tight_layout()
plt.show()