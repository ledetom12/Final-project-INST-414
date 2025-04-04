Documentation:
Project description:
This project explores global poverty and income inequality using a machine learning approach. The dataset used is from Kaggle: Global Poverty and Inequality Dataset by Utkarsh Singh, which includes key socioeconomic indicators like the Gini coefficient, poverty headcount ratios, and income distribution metrics. The goal is to identify patterns in poverty and inequality across countries and build predictive models to better understand which factors most strongly influence poverty rates.  

dependencies :
Python 3.12, pandas, numpy, seaborn, matplotlib, sckikit-learn

setting up enviroment:
in powershell:
python -m venv venv
.\venv\Scripts\activate
 pip install -r finalprojectrequire.txt
Running the project:

There are 4 main files in my project 

finalproject.py:
To run this file in the command line you must run, python finalproject.py
Final Project. After running this code a table will appear in the command line with a bunch of values regarding different variables from the dataset.
To run this file in the command line you must run, python finalproject2.py. After doing this a visualization will appear. 

finalproject2.py:
To run this file in the command line you must run, python finalproject2.py. After running this code different visualizations will be displayed.

featureengineering.py:
To run this file in the command line you must run, python featureengineering.py.After running it, the script outputs a performance table for a linear regression model. This table helps me evaluate how well the features—including the newly created Wealth Gap Index predict poverty rates in different countries
 
decisiontreecodeclassifier.py:
To run this file in the command line you must run, python decisiontreecodeclassifier.py. After doing a trained decision model visualization will appear.

Data processing pipeline/reproducing results:
to run the data processing pipeline, I start by executing finalproject.py, which loads the raw dataset, selects key variables, formats the year column, and normalizes important features like the Gini coefficient. Next, I run featureengineering.py, which creates new variables such as the Wealth Gap Index and evaluates them using a linear regression model. After that, I run decisiontreecode.py to train and test a decision tree model using the cleaned and engineered data. Finally, I execute finalproject2.py to generate visualizations that help interpret patterns in poverty and income inequality across different countries. Running these scripts in order ensures that each step builds on the previous one and the full analysis pipeline is completed smoothly