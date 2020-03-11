#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
f=open("heart.csv","r")
dataset=pd.read_csv(f)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
dataset
from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.2,random_state=0)
X_test,Y_test
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
X_train
from sklearn.tree import DecisionTreeClassifier
decisiontreeclassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
decisiontreeclassifier.fit(X_train,Y_train)
Y_pred = decisiontreeclassifier.predict(X_test)
Y_pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
print("confusion matrix is :")
print(cm)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
var1=metrics.accuracy_score(Y_test, Y_pred)
fout=open("dtree-accuracy.txt","w")
fout.write(str(var1))





