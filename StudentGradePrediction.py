import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
#from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from operator import add
from boruta import BorutaPy

dataFrame = pd.read_csv("student.csv")

#print dataFrame.dtypes
#Splitting dataset to independent and dependent variable
X = dataFrame.iloc[:,:-1]
Y = dataFrame.iloc[:,-1]


#Functions to call Learning Algorithms
def DecisionTreeFunction(X_train, Y_train, X_test, Y_test):
	Tree = DecisionTreeRegressor()
	Tree = Tree.fit(X_train,Y_train)
	return np.array([Tree.score(X_test, Y_test), Tree.score(X_train, Y_train), mean_squared_error(Y_test, Tree.predict(X_test)), mean_squared_error(Y_train, Tree.predict(X_train))])

def KNeighborsFunction(X_train, Y_train, X_test, Y_test):
	KNeighbors = KNeighborsRegressor(n_neighbors = 44, weights = 'distance')
	KNeighbors = KNeighbors.fit(X_train, Y_train)
	return np.array([KNeighbors.score(X_test, Y_test), KNeighbors.score(X_train, Y_train), mean_squared_error(Y_test, KNeighbors.predict(X_test)), mean_squared_error(Y_train, KNeighbors.predict(X_train))])

def SVMFunction(X_train, Y_train, X_test, Y_test):
	SV = SVR(C = 4, epsilon=0.3)
	SV = SV.fit(X_train, Y_train)
	return np.array([SV.score(X_test, Y_test), SV.score(X_train, Y_train), mean_squared_error(Y_test, SV.predict(X_test)), mean_squared_error(Y_train, SV.predict(X_train))])

def RandomForestFunction(X_train, Y_train, X_test, Y_test):
	RF = RandomForestRegressor(n_estimators=4, min_samples_leaf=5)
	RF = RF.fit(X_train,Y_train)
	return np.array([RF.score(X_test, Y_test), RF.score(X_train, Y_train), mean_squared_error(Y_test, RF.predict(X_test)), mean_squared_error(Y_train, RF.predict(X_train))])

#def XGBoostFunction(X_train, Y_train, X_test, Y_test):
#	XGB = XGBRegressor()
#	XGB = XGB.fit(X_train,Y_train)
#	return np.array([XGB.score(X_test, Y_test), XGB.score(X_train, Y_train), mean_squared_error(Y_test, XGB.predict(X_test)), mean_squared_error(Y_train, XGB.predict(X_train))])

def MLPFunction(X_train, Y_train, X_test, Y_test):
	MLP = MLPRegressor()
	MLP = MLP.fit(X_train,Y_train)
	return np.array([MLP.score(X_test, Y_test), MLP.score(X_train, Y_train), mean_squared_error(Y_test, MLP.predict(X_test)), mean_squared_error(Y_train, MLP.predict(X_train))])


#Encoding binary variables. For ex - Male 'M' -> 1 , Female 'F' - 0
labelEncoder = LabelEncoder()
binary_field = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for field in binary_field:
            X[field] = labelEncoder.fit_transform(X[field])


#One hot encoding of categorical values
category_field = ['Mjob','Fjob','reason','guardian']
for category in category_field:
        X_new = pd.get_dummies(X[category])
        X = pd.concat([X,X_new],axis=1)
        del X[category]


X_PCA = pd.concat([X,Y],axis=1)
X_B = X
column_names = list(X_PCA.columns)
print column_names
print len(column_names)
X_PCA = MinMaxScaler().fit_transform(X_PCA)
X_B = MinMaxScaler().fit_transform(X_B)

#Save DF
#np.savetxt(r'/home/rishabh/student1.txt', X_PCA, fmt='%f')
#X_PCA.to_csv(r'/home/rishabh/student2.txt', header=None, index=None, sep=' ', mode='a')


#print column_names
pca = PCA()
pca.fit(X_PCA)
#print pca.explained_variance_ratio_
stdY = np.std(X_PCA[:,43], dtype=np.float64)
print stdY
cov = pca.get_covariance()
X_PCA = np.delete(X_PCA,43,1)

#Print the covariance between each independent variable and dependent variable
for column in range(len(column_names)-1):
	stdXi = np.std(X_PCA[:,column], dtype=np.float64)
	print "%s %f" %(column_names[column], (cov[column][len(column_names)-1])/(stdXi*stdY))

#Remove unnecessary features - PCA
count = 0
remove_col = []
imp_col = []
for column in range(len(column_names)-1):
	stdXi = np.std(X_PCA[:,column], dtype=np.float64)
	if abs((cov[column][len(column_names)-1])/(stdXi*stdY)) < 0.15:
		remove_col.append(column)
		count += 1
	else:
		imp_col.append(column_names[column])
#print count #total number of features removed
print 'Important columns - PCA'
print imp_col #print the important features

X_PCA = np.delete(X_PCA,remove_col,1)

#Remove unnecessary features - Boruta
#print column_names
feat_select = BorutaPy(RandomForestRegressor())
feat_select.fit(X_B,Y)
#print feat_select.n_features_
#print feat_select.ranking_

for column in range(len(column_names)-1):
	print '%s %d' %(column_names[column], feat_select.ranking_[column])
Redundant_col = []
Imp_col_names = set([])
for column in range(len(column_names)-1):
        if feat_select.ranking_[column] > 5:
                Redundant_col.append(column)
        else:
                Imp_col_names.add(column_names[column])
Imp_col_names = list(Imp_col_names)
print 'Important columns based - Boruta'
print Imp_col_names

X_B = np.delete(X_B, Redundant_col,1)

ScoreDecisionTreePCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreDecisionTreeBoruta = np.array([0.0, 0.0, 0.0, 0.0])
ScoreSVMPCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreSVMBoruta = np.array([0.0 , 0.0, 0.0, 0.0])
ScoreKNeighborsPCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreKNeighborsBoruta = np.array([0.0, 0.0, 0.0, 0.0])
ScoreRandomForestPCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreRandomForestBoruta = np.array([0.0, 0.0, 0.0, 0.0])
ScoreXGBPCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreXGBBoruta = np.array([0.0, 0.0, 0.0, 0.0])
ScoreMLPPCA = np.array([0.0, 0.0, 0.0, 0.0])
ScoreMLPBoruta = np.array([0.0, 0.0, 0.0, 0.0])

#List of Machine Learning Models

MLFunctionList = [DecisionTreeRegressor, KNeighborsRegressor, SVR, RandomForestRegressor, XGBRegressor, MLPRegressor]

for count in range(10):
	#Split the data into train and test set
	X_train_PCA, X_test_PCA, Y_train_PCA, Y_test_PCA = train_test_split(X_PCA,Y, test_size = 0.2, random_state = 0)
	X_train_B, X_test_B, Y_train_B, Y_test_B = train_test_split(X_B, Y, test_size = 0.2, random_state = 0)
	#Decision tree
	ScoreDecisionTreePCA += DecisionTreeFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
	ScoreDecisionTreeBoruta += DecisionTreeFunction(X_train_B, Y_train_B, X_test_B, Y_test_B)
	#KNeighbors
	ScoreKNeighborsPCA += KNeighborsFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
	ScoreKNeighborsBoruta += KNeighborsFunction(X_train_B, Y_train_B, X_test_B, Y_test_B) 
	#SVM
	ScoreSVMPCA += SVMFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
	ScoreSVMBoruta += SVMFunction(X_train_B, Y_train_B, X_test_B, Y_test_B)
	#RandomForest
	ScoreRandomForestPCA += RandomForestFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
	ScoreRandomForestBoruta += RandomForestFunction(X_train_B, Y_train_B, X_test_B, Y_test_B)
	#XGB
	#ScoreXGBPCA += XGBoostFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
	#ScoreXGBBoruta += XGBoostFunction(X_train_B, Y_train_B , X_test_B, Y_test_B)
'''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
Neighbors = list(range(1,100))
MSE = []
min_k = 0
for k in Neighbors:
        KNeighbors = neighbors.KNeighborsRegressor(n_neighbors = k, weights = 'distance')
        KNeighbors = KNeighbors.fit(X_train, Y_train)
        MSE.append(mean_squared_error(Y_test, KNeighbors.predict(X_test)))
plt.plot(Neighbors, MSE)
plt.xlabel('Number of neighbors - K')
plt.ylabel('Misclassification error - MSE')
plt.show()
estimators = list(range(1,100))
MSE = []
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
for k in estimators:
        RF = ensemble.RandomForestRegressor(n_estimators=k)
        RF = RF.fit(X_train,Y_train)
	MSE.append(mean_squared_error(Y_test, RF.predict(X_test)))
print RF.score(X_test,Y_test)
plt.plot(estimators, MSE)
plt.xlabel('Estimators')
plt.ylabel('MSE')
plt.show()
'''
ScoreMLPPCA += MLPFunction(X_train_PCA, Y_train_PCA, X_test_PCA, Y_test_PCA)
ScoreMLPBoruta += MLPFunction(X_train_B, Y_train_B, X_test_B, Y_test_B)
print 'Decision tree - PCA (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreDecisionTreePCA/10
print 'Decision tree - Boruta (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreDecisionTreeBoruta/10

print 'KNeighbors - PCA (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreKNeighborsPCA/10
print 'KNeighbors - Boruta (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreKNeighborsBoruta/10

print 'SVM - PCA (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreSVMPCA/10
print 'SVM - Boruta (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreSVMBoruta/10

print 'Random Forest - PCA (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreRandomForestPCA/10
print 'Random Forest - Boruta (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreRandomForestBoruta/10

print 'XGB - PCA (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreXGBPCA/10
print 'XGB - Boruta (R^2 test, R^2 train, MSE test, MSE train)'
print ScoreXGBBoruta/10
print ScoreMLPPCA
print ScoreMLPBoruta
