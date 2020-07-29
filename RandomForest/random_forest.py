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

number_cardiac_case = sum(Y)
number_non_cardiac_cases = n - number_cardiac_case
number_features = len(df.columns)-1

print("Number of cardiac cases is " + str(number_cardiac_case))
print("Number of non-cardiac cases is " + str(number_non_cardiac_cases))


depth = 4
clf = tree.DecisionTreeClassifier(max_depth=depth)
clf = clf.fit(X, Y)
fig = plt.figure()
ax = plt.axes()
# fig, ax = plt.subplots(figsize=(4, 6))
tree.plot_tree(clf.fit(X, Y), max_depth=depth, fontsize=4, class_names=['No Heart Disease', 'Heart Disease'],filled=True)
plt.show()

# clf = tree.DecisionTreeClassifier(max_depth=4)
# clf = clf.fit(X, Y)
# fig, ax = plt.subplots(figsize=(8, 6))
# tree.plot_tree(clf.fit(X, Y), max_depth=4, fontsize=10)
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
# list_depths = list(range(1,20))
# list_miss_rate = []
# list_roc_auc_score =[]
# for depth in list_depths:
#     clf = tree.DecisionTreeClassifier(max_depth=depth, random_state = 42)
#     clf = clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(y_test, predictions)
#     miss_rate = 1-total_accuracy
#     roc_score = roc_auc_score(y_test, predictions)
#     list_miss_rate.append(miss_rate)
#     list_roc_auc_score.append(roc_score)
#
# fig = plt.figure()
# ax = plt.axes()
# plt.title("Decision Tree - AUC score by Tree Depth")
# plt.plot(list_depths, list_roc_auc_score)
# plt.show()
#
# fig = plt.figure()
# ax = plt.axes()
# plt.title("Decision Tree - Misclassification rate by Tree Depth")
# plt.plot(list_depths, list_miss_rate)
# plt.show()
#
#
# list_depths = list(range(1,20))
# list_miss_rate = []
# list_roc_auc_score =[]
# for depth in list_depths:
#     clf = RandomForestClassifier(max_depth=depth, random_state = 42)
#     clf = clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(y_test, predictions)
#     miss_rate = 1-total_accuracy
#     roc_score = roc_auc_score(y_test, predictions)
#     list_miss_rate.append(miss_rate)
#     list_roc_auc_score.append(roc_score)
#
# fig = plt.figure()
# ax = plt.axes()
# plt.title("Random Forest - AUC score by Tree Depth")
# plt.plot(list_depths, list_roc_auc_score)
# plt.show()
#
# fig = plt.figure()
# ax = plt.axes()
# plt.title("Random Forest - Misclassification rate by Tree Depth")
# plt.plot(list_depths, list_miss_rate)
# plt.show()

no_splits=10
cv = KFold(n_splits=no_splits, random_state=42, shuffle=True)

# ### Now doing CV fold to accurately estimate accuracy
accuracy_folds = []
list_error = []

clf = RandomForestClassifier(max_depth=4, random_state = 42)

for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], Y[train_index], Y[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(y_test, predictions)
    miss_rate = 1-total_accuracy
    accuracy_folds.append(total_accuracy)
    list_error.append(miss_rate)

fig = plt.figure()
ax = plt.axes()
plt.title("Random Forest with Depth 4 - Accuracy rate for each fold")
plt.plot(list(range(no_splits)),accuracy_folds)
print("Average Random Forest with Depth 4 accuracy is :" + str(round(np.mean(accuracy_folds),4)*100) + '%')
plt.show()

#################### DECISION TREE ##############
### Now doing CV fold to accurately estimate accuracy
accuracy_folds = []
list_error = []

clf = tree.DecisionTreeClassifier(max_depth=4, random_state = 42)

for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index,:], Y[train_index], Y[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(y_test, predictions)
    miss_rate = 1-total_accuracy
    accuracy_folds.append(total_accuracy)
    list_error.append(miss_rate)

fig = plt.figure()
ax = plt.axes()
plt.title("Decision Tree with Depth 4 - Accuracy rate for each fold")
plt.plot(list(range(no_splits)),accuracy_folds)
print("Average Decision Tree with Depth 4 accuracy is :" + str(round(np.mean(accuracy_folds),4)*100) + '%')
plt.show()

## Finally test model on testing data

clf = RandomForestClassifier(max_depth=4, random_state = 42)
clf.fit(X, Y)

df_test = pd.read_csv('../CleanedData/norm_test.csv')

target_col = ['target']
var_cols = [x for x in df_test.columns if x not in target_col]
X_test = df_test[var_cols].copy()
Y_test = df_test[target_col].values.flatten()
predictions = clf.predict(X_test)
accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(Y_test, predictions)
miss_rate = 1 - total_accuracy
roc = roc_auc_score(Y_test, predictions)
print("Random Forest with Depth 4 testing accuracy is :" + str(round(total_accuracy,4)*100) + '%')
print("Random Forest with Depth 4 roc score is :" + str(round(roc,4)*100) + '%')

x_axis_labels = ['No Heart Disease','Heart Disease']
y_axis_labels = ['No Heart Disease','Heart Disease']
cf_matrix = confusion_matrix(Y_test, predictions)
sns.heatmap(cf_matrix, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Confusion Matrix for Random Forest Testing')
plt.show()

### Check data this for our decision tree as well
clf = tree.DecisionTreeClassifier(max_depth=4, random_state = 42)

clf.fit(X, Y)

df_test = pd.read_csv('../CleanedData/norm_test.csv')

target_col = ['target']
var_cols = [x for x in df_test.columns if x not in target_col]
X_test = df_test[var_cols].copy()
Y_test = df_test[target_col].values.flatten()
predictions = clf.predict(X_test)
accuracy_of_0, accuracy_of_1, total_accuracy = get_accuracy(Y_test, predictions)
miss_rate = 1 - total_accuracy
roc = roc_auc_score(Y_test, predictions)
print("Decision Tree with Depth 4 testing accuracy is :" + str(round(total_accuracy,4)*100) + '%')
print("Decision Tree Forest with Depth 4 roc score is :" + str(round(roc,4)*100) + '%')

x_axis_labels = ['No Heart Disease','Heart Disease']
y_axis_labels = ['No Heart Disease','Heart Disease']
cf_matrix = confusion_matrix(Y_test, predictions)
sns.heatmap(cf_matrix, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Confusion Matrix for Decision Tree Testing')
plt.show()


## Finally plot decision boundary using model trained from 2 most key variables from logistic regression
df = pd.read_csv('../CleanedData/norm_train.csv')
df.fillna(0,inplace=True)

n = len(df)
target_col = ['target']
var_cols = ['thalach','ca']
X = df[var_cols].copy()
Y = df[target_col]

X_test = df_test[['thalach','ca']].copy()
Y_test = df_test[target_col].copy()
Y=Y.values.flatten()

clf = RandomForestClassifier(max_depth=2, random_state = 42)
clf.fit(X, Y)
X = X.to_numpy()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
fig = plt.figure()
ax = plt.axes()

# for idx, clf, tt in zip(product([0], [0]), [clf], ['RandomForest']):

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
ax.set_title('Random Forest Decision Boundary with depth 2')
plt.xlabel('thalach', fontsize=12)
plt.ylabel('ca', fontsize=12)

plt.show()



## Finally show decision boundary for decision tree

clf = tree.DecisionTreeClassifier(max_depth=2, random_state = 42)
clf.fit(X, Y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
fig = plt.figure()
ax = plt.axes()

# for idx, clf, tt in zip(product([0], [0]), [clf], ['RandomForest']):

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
ax.set_title('Decision Tree Decision Boundary with depth 2')
plt.xlabel('thalach', fontsize=12)
plt.ylabel('ca', fontsize=12)

plt.show()

