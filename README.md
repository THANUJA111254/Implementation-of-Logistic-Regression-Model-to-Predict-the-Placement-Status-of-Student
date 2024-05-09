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
### TOP 5 ELEMENTS
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/0a5cda11-f165-4e1b-86ec-5f16a2f1ee09)

![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/01a8cd00-a0ac-49e9-bdc5-116dc5c20f3d)

![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/877b2b6f-3436-47e1-9833-e0f9ad9aa560)

### Data Duplicate:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/9f5231e0-796f-4f94-a0bf-d38784643278)

<br>
<br>
<br>
<br>
<br>
<br>
<br>


### Print Data:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/8ff146fd-7c1b-4323-8bba-b9fedaee4ab1)

### Data-Status:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/8e99861d-c573-4987-9024-596a68482332)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>



### y_prediction array:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/a0cf6d5c-79d4-485e-84ea-e6601f115379)



### Confusion array:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/58525a0d-f694-4ddf-ac84-5b596381c5ef)


### Accuracy Value:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/4f64023f-c3c2-45d2-afca-b68b780f5279)


### Classification Report:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/9a18c485-4dcb-4116-b6d5-f8fdab5de668)

### Prediction of LR:
![image](https://github.com/gauthamkrishna7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/141175025/e0aeefa2-a16d-40cd-b5ab-b1b1a22044f1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
