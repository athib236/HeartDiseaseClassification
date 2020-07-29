######################################################################
# The codes are based on Python3.
# Please install imported packages before using
#
# @version 1.0
# @author Shubhayan
######################################################################

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import neighbors
import numpy as np
from itertools import product
import seaborn as sns
from sklearn.linear_model import RidgeClassifier

def get_accuracy(y_test,predictions):
    result = confusion_matrix(y_test, predictions)
    accuracy_of_0 = result[0, 0] / (result[0, 0] + result[0, 1])
    accuracy_of_1 = result[1, 1] / (result[1, 1] + result[1, 0])
    total_accuracy = (result[0, 0] + result[1, 1]) / sum(sum(result))

    return accuracy_of_0, accuracy_of_1, total_accuracy

df = pd.read_csv('../CleanedData/norm_train.csv')
df.fillna(0,inplace=True)

n = len(df)
target_col = ['target']
var_cols = [x for x in df.columns if x not in target_col]
X = df[var_cols].copy()
Y = df[target_col].values.flatten()
#
# no_splits=10
# cv = KFold(n_splits=no_splits, random_state=42, shuffle=True)
#
# accuracy_apha=[]
# accuracy_folds = []
# list_error = []
# list_alphas = list(np.arange(0,2,0.01))
# for alpha in list_alphas:
#     clf = RidgeClassifier(alpha=alpha)
#     for train_index, test_index in cv.split(X):
#         X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], Y[train_index], Y[test_index]
#         clf.fit(X_train, y_train)
#         predictions = clf.predict(X_test)
#         accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(y_test, predictions)
#         miss_rate = 1-total_accuracy
#         accuracy_folds.append(total_accuracy)
#         list_error.append(miss_rate)
#     mean_accuracy = np.mean(accuracy_folds)
#     accuracy_apha.append(mean_accuracy)
#
# fig = plt.figure()
# ax = plt.axes()
# plt.title("Ridge Regression - Mean Accuracy rate after crossvalidation")
# plt.plot(list_alphas,accuracy_apha)
# plt.show()


### PERFORMING TESTING

optimal_alpha=0.5
clf = RidgeClassifier(alpha=optimal_alpha)
clf.fit(X, Y)
print(clf.coef_)
df_test = pd.read_csv('../CleanedData/norm_test.csv')

target_col = ['target']
var_cols = [x for x in df_test.columns if x not in target_col]
X_test = df_test[var_cols].copy()
Y_test = df_test[target_col].values.flatten()
predictions = clf.predict(X_test)
accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(Y_test, predictions)
print("Testing accuracy is: " + str(total_accuracy))

x_axis_labels = ['No Heart Disease','Heart Disease']
y_axis_labels = ['No Heart Disease','Heart Disease']
cf_matrix = confusion_matrix(Y_test, predictions)
sns.heatmap(cf_matrix, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Confusion Matrix for Ridge Classification Testing')
plt.show()