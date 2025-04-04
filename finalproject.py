import pandas as pd
import seaborn as sea
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score

 
file_path = "pip_dataset.csv"  
df = pd.read_csv(file_path)
 
selected_columns = [
    "country", "year", "headcount_ratio_international_povline", "gini", "welfare_type", "p90_p10_ratio",
    "decile8_thr", "decile9_thr"
]

df_selected = df[selected_columns]

 
df_selected["year"] = df_selected["year"].astype(int)

if 'year' in df_selected.columns:
    df_selected["year"] = df_selected["year"].astype(int)

 
normalize_columns = ["gini", "p90_p10_ratio"]


normalize_columns = [col for col in normalize_columns if col in df_selected.columns]


scaler = MinMaxScaler()
df_selected[normalize_columns] = scaler.fit_transform(df_selected[normalize_columns])

 
df_selected.to_csv("pip_dataset_selected.csv", index=False)

print(df_selected.head())
