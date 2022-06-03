import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt 
df=pd.read_csv("trainml4e.csv")
df=df.dropna()
df['Total_Number']=df['Number_Doses_Week']*df['Number_Weeks_Used']

X=np.array(df.drop(['ID','Crop_Damage','Number_Weeks_Quit'],axis=1))
print(X)
y=np.array(df['Crop_Damage'])
x=preprocessing.scale(X)
print(len(X),len(y))
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.15,random_state=24)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=10)
clf.fit(X_train,y_train)
pred_val=clf.predict(X_val)
print(pred_val)
pred_train=clf.predict(X_train)
print(accuracy_score(y_train,pred_train))
print(accuracy_score(y_val,pred_val))
print (clf.score(X_val,y_val))
print(len(y_val))
df1=pd.read_csv('testml4e.csv')
df1['Total_Number']=df1['Number_Doses_Week']*df1['Number_Weeks_Used']
df1.fillna(df1.median(),inplace=True)

test=np.array(df1.drop(['ID','Number_Weeks_Quit'],axis=1))
pred_test=clf.predict(test)
'''
train_accuracy=[]
test_accuracy=[]
for depth in range(1,20):
    clf=DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train,y_train)
    train_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_val,y_val))
depth=[]
for i in range(1,20):
    depth.append(i)


plt.plot(depth,train_accuracy, label='train percent')
plt.plot(depth,test_accuracy, label='test percent')
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.savefig('accuracy1.png')

'''
df1=pd.DataFrame()
df1['Crop_Damage']=pred_test
df1.to_csv('submit.csv',index=False)
