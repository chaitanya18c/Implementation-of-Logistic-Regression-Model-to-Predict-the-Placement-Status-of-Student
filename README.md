# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression. 

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: CHAITANYA P S
RegisterNumber:  212222230024
*/

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
df.head()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
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
df1
x=df1.iloc[:,:-1]
x
y=df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:\n",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print("Classification Report:\n",classification_report1)
model.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### DataSet Information:
![image](https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/0e822e22-3640-4c3b-bc33-98825910ea72)

### NULL Values:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/b853ab78-135c-4416-b808-4176aaf589e5" height="200">

### Transformed Data:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/2411e0f1-17d3-45d4-91a2-32d239e5a26b" width="500">

### X and Y:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/b5a871a4-fca1-4d6b-aa1c-15940cbc2a41" width="500">
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/7b9f31c1-833f-4a67-9459-f173c35cb79b" width="300">

### Y Predicted:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/41ac6a53-0b8e-4c16-ae93-2f1082912bc0" width="500">

### Accuracy:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/09cd54e1-7502-427d-84f6-117f80d08bd8" width="200">

### Confusion Matrix:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/5649d2e5-cc30-45f3-abf6-80f3e2ab6128" width="200">

### Classification Report:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119393540/9b7b1a88-25dc-45ce-8cd9-e6846eff237c" width="300">

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
