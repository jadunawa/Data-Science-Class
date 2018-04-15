import csv
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

#load the dataset
AH_data=pd.read_csv("2017.csv")
data_clean=AH_data.dropna()

#print(data_clean.dtypes)
#print(data_clean.describe())

#train data
predictors=data_clean[['H','2B','3B','HR','SB','CS','BB','SO','BA','OBP','SLG','TB','GDP','HBP','SH','SF','LOB']]
predictors=data_clean[['RBI','SB']]
targets=data_clean.R
#targets=data_clean.ht

#split data into training/testing
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=0.1)

print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)

#build model

classifier=RandomForestClassifier(n_estimators=30, criterion='gini', oob_score=True)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test,predictions)

#fit extra trees model
print("Extra Trees:")
model=ExtraTreesClassifier()
model.fit(pred_train,tar_train)

print(len(model.predict(pred_test)))


#display importance of each attribute
print(model.feature_importances_)   # what Kim told me to do
print("Accuracy:")

print(classifier.score(pred_test, tar_test))
#print(roc_auc_score())
#area under curve

#compare output to expected