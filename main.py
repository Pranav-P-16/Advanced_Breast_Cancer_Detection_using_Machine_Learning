import numpy,pandas
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = sklearn.datasets.load_breast_cancer()
dataframe = pandas.DataFrame(dataset.data,columns=dataset.feature_names)
#print(dataframe.head())
dataframe['label']=dataset.target
#print(dataframe.tail())
#print(dataframe.shape) #rows and columns info
#print(dataframe.info()) #more info about dataframe
#print(dataframe.isnull().sum()) #to check nullvalues
#print(dataframe.describe()) # for getting statistical info about dataset

# M-malignant, B-benign
dat1=dataframe.groupby('label').mean()
Xaxis=dataframe.drop(columns='label',axis=1)
Yaxis=dataframe['label']

#Splitting training data and testing data
Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xaxis,Yaxis,test_size=0.2,random_state=2) #20% to test data
#print(Xaxis.shape,Xtrain.shape,Xtest.shape)

#Training Logistic Regression model using Training data
model=LogisticRegression(max_iter=10000)
model.fit(Xtrain,Ytrain)

#Accuracy on Trained data
Xtrain_prediction=model.predict(Xtrain)
trained_data_accuracy=accuracy_score(Ytrain,Xtrain_prediction)
print("Accuracy on training data = ",trained_data_accuracy)

#Accuracy on test data
Xtest_prediction=model.predict(Xtest)
tested_data_accuracy=accuracy_score(Ytest,Xtest_prediction)
print("Accuracy on test data = ",tested_data_accuracy)

#Predictive System
input_data=(20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)
input_data_numpy=numpy.asarray(input_data)
reshaped_data=input_data_numpy.reshape(1,-1)
prediction=model.predict(reshaped_data)
if (prediction[0]==0):
    print('Malignant / Cancerous')
elif (prediction[0]==1):
    print('Benign / Non-cancerous')
else:
    print('Error occured!')