# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 


## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GAUTHAM KRISHNA S
RegisterNumber:  212223240036
*/
```
```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement Data:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/cfdb0c4f-6f88-4305-b84c-5f56cf8cae00)

### Salary Data:
![salarydata](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/a7a5765a-7982-42d2-953e-ed535a4542b3)

### Checking the null() function:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/aff04306-b1da-434d-94b0-e80f4b249947)

### Data Duplicate:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/1839e289-58a4-4568-af22-6784a982fa51)

### print Data:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/4886e0af-14f8-4e49-a2b9-88053510a261)

### Data-Status:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/50c361a6-beb6-455f-87ca-0f3e0e15fc94)

### y_prediction array:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/a1fb65df-2916-4908-af35-f0c32cb8ad1e)

### Accuracy Value:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/942a5f56-9e72-43d6-aee8-1b5b4c01542d)

### Confusion array:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/c7ce70be-fc97-44bd-a3c1-abb109e75563)

### Classification Report:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/5e798295-000b-4d42-a56e-6e14f2cfd8bd)

### Prediction of LR:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/79243040-085d-4a2d-827b-bbfccdac6a97)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
