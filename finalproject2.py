import pandas as pd 
import seaborn as sns
from finalproject import df_selected
import matplotlib.pyplot as plt



sns.set(style="whitegrid")

 
top_countries = df_selected.groupby("country")["headcount_ratio_international_povline"].median().sort_values(ascending=False).index[:10]
df_filtered = df_selected[df_selected["country"].isin(top_countries)]

plt.figure(figsize=(14, 6))
sns.boxplot(x="country", y="gini", data=df_filtered, order=top_countries)

 
plt.xticks(rotation=90)

plt.title("Income Inequality (Gini Coefficient) Across Selected Countries")
plt.xlabel("Country")
plt.ylabel("Gini Coefficient (Higher = More Inequality)")

# Show the plot
plt.show()