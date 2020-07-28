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
from sklearn.decomposition import PCA

import statsmodels.api as sm
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
from scipy import stats

## DEFINE ENCODING FUNCTIONS
onehotencoder = OneHotEncoder()
onehotencoder_dropfirst = OneHotEncoder(drop='first')


def hot_encode(df, vars, drop=True):
    for var in vars:
        if drop == True:
            X = onehotencoder_dropfirst.fit_transform(df[var].values.reshape(-1, 1)).toarray().astype(int)
        else:
            X = onehotencoder.fit_transform(df[var].values.reshape(-1, 1)).toarray().astype(int)
        dfOneHot = pd.DataFrame(X, columns=[var + "_" + str(int(i)) for i in range(X.shape[1])])
        df = pd.concat([df, dfOneHot], axis=1)

        df = df.drop([var], axis=1).copy()
    return df


def calculate_vif(data):
    vif_df = pd.DataFrame(columns=['Var', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        r_squared = sm.OLS(y, x).fit().rsquared
        vif = round(1 / (1 - r_squared), 2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by='Vif', axis=0, ascending=False, inplace=False)


def treat_missing(df):
    ## Apart from thal (2 zeros denoting missing) there aren't 0s that wouldn't normally be in data
    ## Replace 2 zero thal with mode =2 (blood flow normal)
    thal_mode = int(df['thal'].mode())
    df['thal'].replace({0: thal_mode}, inplace=True)
    return df


df = pd.read_csv('datasets_33180_43520_heart.csv')

############## TREAT MISSING VALUES IF ANY  ####################################################

# Check for NaNs, returns False
anyNan = df.isnull().values.any()

zeros = (df == 0).astype(int).sum()

df = treat_missing(df)


############## NORMALIZE (MinMax) Categorical Ordinal Variables ###################

# TODO
# Reorder categorical var for slope so that it is ordinal
#  slope: the slope of the peak exercise ST segment
# -- Value 1: upsloping
# -- Value 2: flat
# -- Value 3: downsloping

def normalize_ordinals(df):
    cat_ordinals = ['slope', 'ca']
    scaler = MinMaxScaler()
    for var in cat_ordinals:
        x = df[var]
        x_scaled = scaler.fit_transform(x.values.reshape(-1, 1))
        # sns.distplot(xt_scaled)
        # plt.show()
        df[var] = x_scaled
    return df

df = normalize_ordinals(df)

## Box - whisker plots for continuous vars
# ax = sns.boxplot(x=df['trestbps'])
# ax = sns.boxplot(x=df['chol'])
# ax = sns.boxplot(x=df['thalach'])
# ax = sns.boxplot(x=df['oldpeak'])
# ax = sns.boxplot(x=df['age'])

## Remove outliers for cont vars for normalizing data

## Oldpeak will be normalized so remove outliers
df.oldpeak = np.where(df.oldpeak > 4, 4, df.oldpeak)

# ax = sns.boxplot(x=df['oldpeak'])

######## BEGIN NORMALIZING (MINMAX) CONTINUOUS VARIABLES ####################

def normalize_cont_vars(df):
    df_norm = df.copy()

    df_norm.trestbps = np.where(df_norm.trestbps > 170, 170, df_norm.trestbps)
    # df_std.trestbps.loc[df_std.trestbps > 170]  = 170
    # ax = sns.boxplot(x=df_std['trestbps'])

    df_norm.chol = np.where(df_norm.chol > 360, 360, df_norm.chol)
    # df_std.chol.loc[df_std.chol > 360]  = 360
    # ax = sns.boxplot(x=df_std['chol'])

    df_norm.thalach = np.where(df_norm.thalach < 90, 90, df_norm.thalach)
    # ax = sns.boxplot(x=df_std['thalach'])

    # df_norm.oldpeak = np.where(df_norm.oldpeak > 4, 4, df_norm.oldpeak)
    # ax = sns.boxplot(x=df_std['oldpeak'])

    scaler = MinMaxScaler()
    cont_vars = ['trestbps', 'chol', 'thalach', 'age', 'oldpeak']
    for var in cont_vars:
        x = df_norm[var]
        x_scaled = scaler.fit_transform(x.values.reshape(-1, 1))
        df_norm[var] = x_scaled
    return df_norm


df_norm = normalize_cont_vars(df)


########## END NORMALIZING CONTINUOUS VARIABLES ###############################


######## BEGIN STANDARDIZING (STDNORM) CONTINUOUS VARIABLES ####################

## Outliers are not required to be removed for standardizing data

## Box cox transform

def standardize_cont(df):
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

    cont_vars = ['trestbps', 'chol', 'thalach', 'age']  # oldpeak could not be transformed so should be scaled
    scaler = StandardScaler()
    for var in cont_vars:
        x = df_std[var]
        x_min = min(x[x > 0])
        x = x + x_min / 2
        xt, _ = stats.boxcox(x)
        xt_scaled = scaler.fit_transform(xt.reshape(-1, 1))
        # sns.distplot(xt_scaled)
        # plt.show()
        df_std[var] = xt_scaled

    scaler = MinMaxScaler()
    df_std['oldpeak'] = scaler.fit_transform(df_std['oldpeak'].values.reshape(-1, 1))
    return df_std


df_std = standardize_cont(df)

###### FINISHED STANDARDIZING CONTINUOUS VARIABLES ##################

########### BEGIN ONE HOT ENCODING NORMALIZED (MINMAX) DATA #################

df_norm_drop = hot_encode(df_norm, ['cp', 'restecg', 'thal'])
df_norm = hot_encode(df_norm, ['cp', 'restecg', 'thal'], drop=False)

## Check VIF
# X_cols = [x for x in df_norm_drop.columns if x not in ['target'] ]
# df_X = df_norm_drop[X_cols].copy()
# calculate_vif(df_X)

########### BEGIN ONE HOT ENCODING STANDARDIZED (STDNORM) DATA #################

df_std_drop = hot_encode(df_std, ['cp', 'restecg', 'thal'])
df_std = hot_encode(df_std, ['cp', 'restecg', 'thal'], drop=False)

## Check VIF
X_cols = [x for x in df_std_drop.columns if x not in ['target']]
df_X = df_norm_drop[X_cols].copy()
calculate_vif(df_X)


##### CENTER THE CATEGORICAL VARIABLES IN STANDARDIZED DATA TO BE FROM [-.5,.5] by subtracting -.5

def center_std_categoricals(df_std):
    std_cont_vars = ['trestbps', 'chol', 'thalach', 'age', 'target']

    # # FOR ONE HOT DROP VERSION
    # cat_cols = [x for x in df_std_drop.columns if x not in std_cont_vars]
    # for var in cat_cols:
    #     x = df_std_drop[var]-0.5
    #     df_std_drop[var] = x

    # FOR WITHOUT DROP VERSION
    cat_cols = [x for x in df_std.columns if x not in std_cont_vars]
    for var in cat_cols:
        x = df_std[var] - 0.5
        df_std[var] = x
    return df_std


df_std_drop = center_std_categoricals(df_std_drop)
df_std = center_std_categoricals(df_std)

### PCA
target_col = ['target']
var_cols = [x for x in df_norm_drop.columns if x not in target_col]
X = df_norm_drop[var_cols].copy()
# Y = df_norm_drop[target_col].values.flatten()

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
df_pca_2comp = pd.concat([principalDf, df[['target']]], axis=1)


################# SPLIT DATA ####################
# df_encoded = hot_encode(df, ['cp', 'restecg', 'thal'], drop=False)
def split_data(df):
    target_col = ['target']
    var_cols = [x for x in df.columns if x not in target_col]
    X = df[var_cols].copy()
    Y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    return df_train, df_test


df_train_std_drop, df_test_std_drop = split_data(df_std_drop)
df_train_std, df_test_std = split_data(df_std)

df_train_norm_drop, df_test_norm_drop = split_data(df_norm_drop)
df_train_norm, df_test_norm = split_data(df_norm)

########### OUTPUT CLEANED DATA ######################################

df_train_std_drop.to_csv('CleanedData/std_onehot_drop_train.csv', index=False)
df_test_std_drop.to_csv('CleanedData/std_onehot_drop_test.csv', index=False)

df_train_std.to_csv('CleanedData/std_train.csv', index=False)
df_test_std.to_csv('CleanedData/std_test.csv', index=False)

df_train_norm_drop.to_csv('CleanedData/norm_onehot_drop_train.csv', index=False)
df_test_norm_drop.to_csv('CleanedData/norm_onehot_drop_test.csv', index=False)

df_train_norm.to_csv('CleanedData/norm_train.csv', index=False)
df_test_norm.to_csv('CleanedData/norm_test.csv', index=False)

## Not creating test and train for PCA data unless required
df_pca_2comp.to_csv('CleanedData/pca_2comp.csv', index=False)
