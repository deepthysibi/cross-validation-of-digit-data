import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



#sample of folding
fold=KFold(n_splits=4)#default n_split is 5
list1=[1,3,5,6,1,8,9,5,6,7,9,10]      #x=train data set and y=test data set
for x,y in fold.split(list1):
    print(x,y)
'''
expected output is:[ 3  4  5  6  7  8  9 10 11] [0 1 2]
                   [ 0  1  2  6  7  8  9 10 11] [3 4 5]
                   [ 0  1  2  3  4  5  9 10 11] [6 7 8]
                   [0 1 2 3 4 5 6 7 8] [ 9 10 11]
'''

#cross validation on digit data set
digits=load_digits()
print(digits.keys())

print(digits.data.shape)

print(digits)

print(digits.data[100])

print(digits.data[1])

print(digits.target[:10])

print(digits.target[:100])

plt.matshow(digits.images[0])

plt.matshow(digits.images[28])

plt.matshow(digits.images[9])

print(digits.target[15])

print(digits.target[129])

#create an instence of stratifiedKFold with 4 folds
fold=StratifiedKFold(n_splits=4)

#create a function to get score of each model
def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)

#create empty list to append score from each fold of model
log_score=[]
svm_score=[]
rf_score=[]

for train_index,test_index in fold.split(digits.data,digits.target):
    x_train,x_test,y_train,y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    log_score.append(get_score(LogisticRegression(),x_train,x_test,y_train,y_test))
    svm_score.append(get_score(SVC(),x_train,x_test,y_train,y_test))
    rf_score.append(get_score(RandomForestClassifier(random_state=34),x_train,x_test,y_train,y_test))

print(log_score)
#we get 4 score from each fold

print(svm_score)

print(rf_score)

print(np.mean(log_score))
print(np.mean(svm_score))
print(np.mean(rf_score))

#by comparing score we can select the best model


#shortcut method of crossvalidation using cross_val_score

from sklearn.model_selection import cross_val_score

#here we are not specifing the fold hence default fold split is 5
log_score=cross_val_score(LogisticRegression(),X=digits.data,y=digits.target)
svm_score=cross_val_score(SVC(),X=digits.data,y=digits.target)
rf_score=cross_val_score(RandomForestClassifier(random_state=43),X=digits.data,y=digits.target)

print(log_score)
print(svm_score)
print(rf_score)

print(np.mean(log_score))
print(np.mean(svm_score))
print(np.mean(rf_score))

#using folds in cross validation

fold=StratifiedKFold(n_splits=4)

#here we specifing the fold
log_score=cross_val_score(LogisticRegression(),X=digits.data,y=digits.target,cv=fold)
svm_score=cross_val_score(SVC(),X=digits.data,y=digits.target,cv=fold)
rf_score=cross_val_score(RandomForestClassifier(random_state=43),X=digits.data,y=digits.target,cv=fold)

print(log_score)
print(svm_score)
print(rf_score)

print(np.mean(log_score))
print(np.mean(svm_score))
print(np.mean(rf_score))




