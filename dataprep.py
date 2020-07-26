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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
import seaborn as sns
import numpy as np
sns.set(style="whitegrid")
from scipy import stats


def get_accuracy(y_test,predictions):
    result = confusion_matrix(y_test, predictions)
    accuracy_of_0 = result[0, 0] / (result[0, 0] + result[0, 1])
    accuracy_of_1 = result[1, 1] / (result[1, 1] + result[1, 0])
    total_accuracy = (result[0, 0] + result[1, 1]) / sum(sum(result))

    return accuracy_of_0, accuracy_of_1, total_accuracy

df = pd.read_csv('datasets_33180_43520_heart.csv')

# Check for NaNs, returns False
anyNan = df.isnull().values.any()

zeros = (df == 0).astype(int).sum()

## Apart from thal (2 zeros denoting missing) there aren't 0s that wouldn't normally be in data

## Replace 2 zero thal with mode =2 (blood flow normal)
thal_mode = int(df['thal'].mode())
condition_thal_zero = df['thal']==0
df['thal'].replace({0:thal_mode},inplace=True)

## Reorder categorical var for slope so that it is ordinal
##  slope: the slope of the peak exercise ST segment
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping

## Box - whisker plots for continuous vars
# ax = sns.boxplot(x=df['trestbps'])
# ax = sns.boxplot(x=df['chol'])
# ax = sns.boxplot(x=df['thalach'])
# ax = sns.boxplot(x=df['oldpeak'])
# ax = sns.boxplot(x=df['age'])

## Remove outliers for cont vars for normalizing data

df_norm = df.copy()

df_norm.trestbps = np.where(df_norm.trestbps > 170, 170, df_norm.trestbps)
# df_std.trestbps.loc[df_std.trestbps > 170]  = 170
# ax = sns.boxplot(x=df_std['trestbps'])

df_norm.chol = np.where(df_norm.chol > 360, 360, df_norm.chol)
# df_std.chol.loc[df_std.chol > 360]  = 360
# ax = sns.boxplot(x=df_std['chol'])

df_norm.thalach = np.where(df_norm.thalach < 90, 90, df_norm.thalach)
# ax = sns.boxplot(x=df_std['thalach'])

df_norm.oldpeak = np.where(df_norm.oldpeak > 4, 4, df_norm.oldpeak)
# ax = sns.boxplot(x=df_std['oldpeak'])

scaler = StandardScaler()
cont_vars = ['trestbps','chol','thalach','age','oldpeak']
categorical

## Outliers are not required to be removed for standardizing data

## Box cox transform

df_std = df.copy()

## Created probplots for BoxCox Transforms
# var = 'age'
# fig = plt.figure()
# fig.suptitle("Box-Cox transformation for " + var, fontsize=14)
# ax1 = fig.add_subplot(211)
# x = df_std[var]
# prob = stats.probplot(x, dist=stats.norm, plot=ax1)
# ax1.set_xlabel('')
# ax1.set_title('Probplot against normal distribution')
#
# ax2 = fig.add_subplot(212)
# x_min = min(x[x>0])
# x = x + x_min/2
# xt, _ = stats.boxcox(x)
# prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
# ax2.set_title('Probplot after Box-Cox transformation')
# # df_std['var']=xt
# plt.show()

cont_vars = ['trestbps','chol','thalach','age'] # oldpeak could not be transformed so should be scaled
scaler = StandardScaler()
for var in cont_vars:
    x = df_std[var]
    x_min = min(x[x > 0])
    x = x + x_min / 2
    xt, _ = stats.boxcox(x)
    xt_scaled = scaler.fit_transform(xt.reshape(-1, 1))
    # sns.distplot(xt_scaled)
    # plt.show()
    df_std[var]=xt_scaled

scaler = MinMaxScaler()
df_std['oldpeak'] = scaler.fit_transform(df_std['oldpeak'])

# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)


## Encoding
onehotencoder = OneHotEncoder()
onehotencoder_dropfirst = OneHotEncoder(drop='first')

def hot_encode(df,vars,drop=True):
    for var in vars:
        if drop == True:
            X = onehotencoder_dropfirst.fit_transform(df[var].values.reshape(-1,1)).toarray().astype(int)
        else:
            X = onehotencoder.fit_transform(df[var].values.reshape(-1, 1)).toarray().astype(int)
        dfOneHot = pd.DataFrame(X, columns = [var + "_" +str(int(i)) for i in range(X.shape[1])])
        df = pd.concat([df, dfOneHot], axis=1)

        df= df.drop([var], axis=1).copy()
    return df

def calculate_vif(data):
    vif_df = pd.DataFrame(columns = ['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y,x).fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)

df_drop = hot_encode(df,['cp','restecg','thal'])
df = hot_encode(df,['cp','restecg','thal'],drop=False)
# df = hot_encode(df,'restecg')

# df_cp = df[['cp_0','cp_1','cp_2','cp_3','target']]

## Check VIF
X_cols = [x for x in df.columns if x is not 'target']
df_X = df[X_cols].copy()
calculate_vif(df_X)


## Normalized version


### Standardized version


### PCA


#
# # Replace and
# df.fillna(0,inplace=True)
#
#
# names = pd.read_csv('spambase/names.csv')
# df.columns=list(names['names'])
# n = len(df)
#
# X = df.iloc[:,:-1].copy()
# Y = df.iloc[:,-1].copy()
#
# number_spam = sum(Y)
# number_non_spam = n-number_spam
# number_features = len(df.columns)-1
#
# depth = 3
# clf = tree.DecisionTreeClassifier(max_depth=depth)
# clf = clf.fit(X, Y)
# fig = plt.figure()
# ax = plt.axes()
# # fig, ax = plt.subplots(figsize=(4, 6))
# tree.plot_tree(clf.fit(X, Y), max_depth=depth, fontsize=6, class_names=['Not Spam', 'Spam'],filled=True)
# plt.savefig('tree_high_dpi', dpi=100)
# plt.show()



# clf = tree.DecisionTreeClassifier(max_depth=4)
# clf = clf.fit(X, Y)
# fig, ax = plt.subplots(figsize=(8, 6))
# tree.plot_tree(clf.fit(X, Y), max_depth=4, fontsize=10)
# plt.show()

#
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
