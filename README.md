## Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.finally execute the program and display the output
 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: APARNA RB
RegisterNumber:  212222220005
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```
![375967605-cf8a7981-4a78-4139-a4f7-7e805a165801](https://github.com/user-attachments/assets/31f667cc-58bc-42ca-a075-c60748037442)
```
dataset.tail()
```
![375967753-90fa2593-5070-44c2-aff8-01e6e8e2f215](https://github.com/user-attachments/assets/f54a5b24-9bd8-4377-9879-64c6e68939be)
```
dataset.info()
```
![375967791-1756b603-2ad3-4f09-9937-692da74b0e1a](https://github.com/user-attachments/assets/0762403c-4bf7-44f9-922a-0a42d059f8cf)
```
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/fa4580ad-0b63-473a-b972-729d5bb40048)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![375968065-33f432c8-5372-4098-b7b9-539ca5aaacd6](https://github.com/user-attachments/assets/abf1abca-519b-4fa9-bdd2-d45dee104714)
```
dataset.info()
```
![375968418-1ef044f5-3c79-4386-a19f-c5f289973e8e](https://github.com/user-attachments/assets/c413d1b7-1fb9-4732-8b20-9fb04ca3db65)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
```
![375969419-8476198b-dfb7-4a5a-aff0-0d1fb6067020](https://github.com/user-attachments/assets/944ed0e4-db0b-419b-b60e-74084dbfb21f)
```
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
```
![375969461-580075e1-1e6c-4d5c-b394-8f3f10b43de8](https://github.com/user-attachments/assets/95401aaa-f323-4398-be6b-be66326cc4b1)
```
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
```
![375969630-5732a1c2-fc6b-4be9-8d2f-7a64d323f5e6](https://github.com/user-attachments/assets/f25ef57d-10aa-434e-8699-4e7d9478f29e)
```
accuracy=accuracy_score(y_test, y_pred)
accuracy
```
![375969477-09d1a12a-cee3-4f16-bf92-1de7f83bb68f](https://github.com/user-attachments/assets/6231f93a-726b-447f-b21e-1671a5021170)


##
Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
