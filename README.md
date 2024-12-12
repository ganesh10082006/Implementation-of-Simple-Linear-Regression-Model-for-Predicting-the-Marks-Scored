# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GANESH G.
RegisterNumber: 212223230059

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('/student_scores.csv')
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
ypred=regressor.predict(x_test)
print(ypred)
print(y_test)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color='orange')
plt.title("Hours vs scores (Trainin Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color='yellow')
plt.title("Hours vs scores (Trainin Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,ypred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,ypred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
![image](https://github.com/ganesh10082006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151981672/04dbad5b-a779-4c24-b5b3-3481bab5b690)

![image](https://github.com/ganesh10082006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151981672/a10a0e81-7be6-4ca5-b3f1-4c232ea99abf)

![image](https://github.com/ganesh10082006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151981672/db89b7bf-4a52-48dd-a679-fd21fd48f99e)

![image](https://github.com/ganesh10082006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151981672/998ba5a5-fe10-4052-ad5f-8293641bdd9a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
