import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import finalproject
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

 
file_path = "pip_dataset_selected.csv"   
df_final = pd.read_csv(file_path)

 
df_numeric = df_final.drop(columns=["country", "welfare_type"])

 
df_ind = df_numeric.fillna(df_numeric.median(numeric_only=True))

 
X = df_ind.drop(columns=["headcount_ratio_international_povline"])  
y = df_ind["headcount_ratio_international_povline"]   

 
y_class = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])  # Convert poverty rates into 3 categories

# Split data into training and testing sets (80% train, 20% test)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
 
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train_cls, y_train_cls)

 
y_pred_dt = decision_tree_classifier.predict(X_test_cls)

 
accuracy_dt = accuracy_score(y_test_cls, y_pred_dt)
cv_scores_dt = cross_val_score(decision_tree_classifier, X, y_class, cv=5, scoring='accuracy')

# Store Decision Tree results
decision_tree_results = {
    "Accuracy": accuracy_dt,
    "Cross-Validation Accuracy Mean": np.mean(cv_scores_dt),
    "Cross-Validation Accuracy Std": np.std(cv_scores_dt)}

plt.figure(figsize=(12, 6))
plot_tree(decision_tree_classifier, filled=True, feature_names=X.columns, class_names=["Low", "Medium", "High"])
plt.title("Decision Tree Classifier Visualization")
plt.show()