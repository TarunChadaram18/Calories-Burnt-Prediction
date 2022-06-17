# -*- coding: utf-8 -*-
"""prediction-of-calories-burnt-using-xgboost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UTm7HtPlHp-nk95MmJ8ok44I9VbJ4ntM

**IMPORTING LIBRARIES**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from google.colab import files

"""**LOADING THE DATA & PREPROCESSING THE DATA**"""

uploaded=files.upload()
calories_data = pd.read_csv("calories.csv")

calories_data.head()

uploaded=files.upload()
exercise_data = pd.read_csv("exercise.csv")

exercise_data.head()

"""**Inference:**

* We can observe an Indirect relationship between heart rate & body temperature with respect to the excercise the individual does.

**COMBINING THE EXCERCISE DATA & CALORIES DATA**
"""

combined_data = pd.concat([exercise_data,calories_data['Calories']], axis=1)

combined_data.head()

#checking the number of rows & columns
combined_data.shape

"""**Hence our combined_data has 15,000 people with 9 different features**"""

#getting the information about the data most importantly to know about any missing values
combined_data.info()

#checking the missing values
combined_data.isnull().sum()

"""**ANALYSING THE DATA**"""

#To get stastical measures about the data
combined_data.describe()

"""**Important inference**
* Heart rate & body temperature is more when the person is doing excercise

**VISUALIZATION OF DATA**
"""

sns.set()

#inorder to find the distribution of males & females
#plotting the gender column in count plot
sns.countplot(combined_data['Gender'])

"""**From the above plot, we can observe that the data is evenly distributed(almost 7000) for both males and females.**"""

#inorder to find the distribution of age column
sns.distplot(combined_data['Age'])

"""**From the above plot, we can observe that the Age column has the mean value of 42 and the data is positively skewed & most of the datapoints are in between 20 to 50.**
    
"""

#inorder to find the distribution of height column
sns.distplot(combined_data['Height'])

"""**Hence,from the above plot we can observe that the height data is normally distributed with mean of 174cm.**"""

#inorder to find the distribution of weight column
sns.distplot(combined_data['Weight'])

"""**Hence,from the above plot we can observe that the weight data is also normally distributed with mean of 74 kg**"""

sns.distplot(combined_data['Duration'])

"""**Hence, from the above plot we can observe that the duration data is a low peaked data with mean of 15mins**"""

sns.distplot(combined_data['Heart_Rate'])

"""**Hence,from the above plot we can observe that the heart rate column is normally distributed with mean of 95 heart beats with several outliers.**"""

sns.distplot(combined_data['Body_Temp'])

"""**Hence,from the above plot we can observe that the temperature column is negatively skewed with mean of 40.**

**TO FIND THE CORRELATION IN THE DATA**

Duration & calories are directly proportional=positively correlated
"""

correlation = combined_data.corr()
correlation

"""**Construction of heat map for the understanding of correlation**"""

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Purples')

"""**CONVERSION OF TEXT DATA TO NUMERICAL VALUES**"""

combined_data.replace({'Gender':{'male':0,'female':1}},inplace=True)
combined_data.head()

"""**SEPARATION OF FEATURES AND TARGET**"""

X=combined_data.drop(['User_ID','Calories'],axis=1)
Y=combined_data['Calories']

print(X)

print(Y)

"""**SPLIT THE DATA INTO TRAINING DATA & TEST DATA**"""

X_train,X_test,Y_train,Y_test = train_test_split(X.values,Y.values,test_size=0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

"""Hence, we can observe that 
* X contains the orignal amount of data which is 15000
* X_train contains 80% of the data which is 12000
* X_test contains 20% of the data which is 3000

**MODEL IMPLEMENTATION : XGBoost Regressor**
"""

import time

start_time = time.clock()

model = XGBRegressor()
model.fit(X_train,Y_train)

print(time.clock() - start_time, "seconds")

calories_burnt_prediction = model.predict(X_test)

print(calories_burnt_prediction)

MAE = metrics.mean_absolute_error(Y_test, calories_burnt_prediction)

print("Mean Absolute Error = ",MAE)

"""**BUILDING A PREDICTIVE SYSTEM**
* Building a predictive system inorder to find the calories burnt for the first individual from the dataset
"""

import numpy as np
input_data = [1,31,148.0,50.0,8.0,84.0,39.5]

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

"""To perform for Linear Regression"""

start_time = time.clock()
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
print(time.clock() - start_time, "seconds")

training_data_prediction=reg.predict(X_train)

MAE_linear_reg_train_data=metrics.mean_absolute_error(Y_train,training_data_prediction)
MAE_linear_reg_train_data

test_data_prediction=reg.predict(X_test)
print(test_data_prediction)

MAE_linear_reg_test_data=metrics.mean_absolute_error(Y_test,test_data_prediction)
MAE_linear_reg_test_data

import numpy as np
input_data = [1,31,148.0,50.0,8.0,84.0,39.5]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = reg.predict(input_data_reshaped)
print(prediction[0])

"""Random Forest Regression"""

start_time = time.clock()
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=300, max_depth=13, random_state=2)
forest_model.fit(X_train,Y_train)
print(time.clock() - start_time, "seconds")

forest_prediction=forest_model.predict(X_test)
forest_prediction
MAE_forest=metrics.mean_absolute_error(Y_test,forest_prediction)
MAE_forest

input_data = [1,31,148.0,50.0,8.0,84.0,39.5]

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = forest_model.predict(input_data_reshaped)
print(prediction[0])