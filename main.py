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
Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xaxis,Yaxis,test_size=0.2,random_state=2)
print(Xaxis.shape,Xtrain.shape,Xtest.shape)
