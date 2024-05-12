# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard libraries.
2Upload the dataset and check for any null or duplicated values using .isnull() and.duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUJITHRA B K N
RegisterNumber: 212222230153

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
print(df.head())

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
print(df1.head())

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
print(df1)

x=df1.iloc[:,:-1]
print(x)

y=df1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
ORIGINAL DATA

![Screenshot 2024-03-21 092010](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/f7020029-e5bd-40d7-bf9a-2e6c4a6cc526)


AFTER REMOVING

![Screenshot 2024-03-21 092101](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/359fcd65-9fba-46d5-8091-d19db6852432)


NULL DATA

![Screenshot 2024-03-21 092111](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/d83f5dd4-0e84-41ca-a104-ac31e13f89cd)


LABEL ENCODER

![Screenshot 2024-03-21 092144](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/f18f9a1d-2a46-4897-b4eb-8bb2f02c95b0)


X VALUES

![Screenshot 2024-03-21 092153](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/60c4563a-95d2-4837-b07c-59fec2b41e13)


Y VALUES

![Screenshot 2024-03-21 092204](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/6a7aceb1-973f-4d79-a277-d6e8564f35a6)


Y_Prediction

![Screenshot 2024-03-21 092215](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/2c06fe35-06b9-4d57-bda9-78a8f2153d65)


ACCURACY

![Screenshot 2024-03-21 092225](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/87d97918-98c3-4ada-9a88-5ab4bc28a632)


CONFUSION MATRIX

![Screenshot 2024-03-21 092246](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/2daf7b84-2da6-4ab5-b394-1d39493079be)


CLASSIFICATION

![Screenshot 2024-03-21 092257](https://github.com/sujithrabkn/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477857/a7a15675-3a7c-4ba2-b4d3-7b81dfb525df)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
