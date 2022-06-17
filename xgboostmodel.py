from fileinput import filename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import pickle

calories_data = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")
combined_data = pd.concat([exercise_data,calories_data['Calories']], axis=1)
combined_data.replace({'Gender':{'male':0,'female':1}},inplace=True)
X=combined_data.drop(['User_ID','Calories'],axis=1)
Y=combined_data['Calories']

X_train,X_test,Y_train,Y_test = train_test_split(X.values,Y.values,test_size=0.2,random_state=2)

model = XGBRegressor()
model.fit(X_train,Y_train)

filename="calories_prediction_model.pkl"
pickle.dump(model, open(filename, "wb"))