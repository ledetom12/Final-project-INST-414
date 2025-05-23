import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
 
df = pd.read_csv("pip_dataset_selected.csv")
df_numeric = df.drop(columns=["country", "welfare_type"])
df_cleaned = df_numeric.fillna(df_numeric.median(numeric_only=True))

 
X = df_cleaned.drop(columns=["headcount_ratio_international_povline"])
y = df_cleaned["headcount_ratio_international_povline"]

 
y_class = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])

 
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
 
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],      
    'criterion': ['gini', 'entropy']       
}

 
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

 
best_tree = grid_search.best_estimator_

 
y_pred = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(best_tree, X, y_class, cv=5, scoring='accuracy')

# Print all evaluation statistics
print("\nDecision Tree Tuning Results:")
print("Best Parameters:", grid_search.best_params_)
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Cross-Validation Accuracy Mean: {np.mean(cv_scores):.4f}")
print(f"Cross-Validation Accuracy Std: {np.std(cv_scores):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
plt.figure(figsize=(16, 8))
plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=["Low", "Medium", "High"])
plt.title("Tuned Decision Tree Classifier Visualization")
plt.tight_layout()
plt.show()

 
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Tuned Decision Tree")
plt.grid(False)
plt.show()