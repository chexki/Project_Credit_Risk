# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:24:34 2018

@author: Chexki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%  Data Loading

loan_df= pd.read_csv(r'D:\DATA SCIENCE\Python\Assignment 1\XYZCorp_LendingData.txt',header=0,delimiter='\t',low_memory=False)
print(loan_df)

#%%
# Generating a dummy dataset

loan_df1 = pd.DataFrame.copy(loan_df)
print(loan_df1.describe(include= 'all'))

#%% Feature Selection

# Some of the variables are not helpful in order to build a predictive model, hence dropping.

loan_df1.drop(['id','member_id','funded_amnt_inv','grade','emp_title','pymnt_plan','desc','title','addr_state',
            'inq_last_6mths','mths_since_last_record','initial_list_status','mths_since_last_major_derog','policy_code','application_type'
            ,'annual_inc_joint','dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m'
            ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
            'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1,inplace=True)

print(loan_df1.head())

#%%
# Checking if missing values are present.
loan_df1.isnull().sum()

#%%
print(loan_df1.dtypes)

#%%
# Imputing categorical missing data with mode value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'issue_d','purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d']
for x in colname1[:]:
    loan_df1[x].fillna(loan_df1[x].mode()[0],inplace=True)
    
loan_df1.isnull().sum()

#%%
colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    loan_df1[x].fillna(loan_df1[x].mean(),inplace=True)
    
loan_df1.isnull().sum()

#%%
print(loan_df1.shape)
loan_df1.describe()    

########################################################################################
#%%         OUTLIERS
#%%   Exploratory analysis : Graphical representation to know more about data,
#     Using boxplot.

loan_df1.boxplot()
plt.xticks(rotation=90) 
plt.show()

#%%

# Handling Extreme Outliers by replacing the with mean of respective features. 

#%%
loan_df1.boxplot(column= 'dti')
plt.show()
# Histogram
loan_df1['dti'].hist(bins=20)
loan_df1['dti'].describe()
#%%
q1_dti = loan_df1['dti'].quantile(0.25)
q3_dti = loan_df1['dti'].quantile(0.75)
iqr_dti = q3_dti-q1_dti                                    # Interquantile range
low_dti = q1_dti-1.5*iqr_dti                               # Acceptable range
high_dti = q3_dti+1.5*iqr_dti

# meeting the acceptable range

loan_df1_include_dti = loan_df1.loc[(loan_df1['dti']> low_dti) & \
                                     (loan_df1['dti']< high_dti)]
# not meeting the acceptable range
loan_df1_except_dti = loan_df1.loc[(loan_df1['dti']<= low_dti) | \
                                    (loan_df1['dti']>= high_dti)]

print(loan_df1_include_dti.shape)   # no. of acceptable observations
print(loan_df1_except_dti.shape)    # no. of outliers
print(high_dti,low_dti)

#finding mean of the acceptable range
dti_mean = loan_df1_include_dti.dti.mean() 
print(dti_mean)

#imputing outliers values with mean value
loan_df1_except_dti.dti = dti_mean
# concatenating both acceptable and except_dtied range
loan_df1 = pd.concat([loan_df1_include_dti,loan_df1_except_dti])
loan_df1.shape

#%%
loan_df1.boxplot(column= 'dti')
plt.show()
loan_df1['dti'].hist(bins=20)
loan_df1['dti'].describe()
########################################################################################
#%%
loan_df1.boxplot(column= 'revol_util')
plt.show()
# Histogram
loan_df1['revol_util'].hist(bins=20)
loan_df1['revol_util'].describe()
#%%
# treating outliers
q1_ru = loan_df1['revol_util'].quantile(0.25)
q3_ru = loan_df1['revol_util'].quantile(0.75)
iqr_ru = q3_ru-q1_ru                                    # Interquantile range
low_ru = q1_ru-1.5*iqr_ru                               # Acceptable range
high_ru = q3_ru+1.5*iqr_ru

# meeting the acceptable range
loan_df1_include_ru = loan_df1.loc[(loan_df1['revol_util']> low_ru) & \
                                     (loan_df1['revol_util']< high_ru)]
# not meeting the acceptable range
loan_df1_except_ru = loan_df1.loc[(loan_df1['revol_util']<= low_ru) | \
                                    (loan_df1['revol_util']>= high_ru)]

print(loan_df1_include_ru.shape)   # no. of acceptable observations
print(loan_df1_except_ru.shape)    # no. of outliers
print(high_ru,low_ru)

#finding mean of the acceptable range
revol_util_mean = loan_df1_include_ru.revol_util.mean() 
print(revol_util_mean)

#imputing outliers values with mean value
loan_df1_except_ru.revol_util = revol_util_mean

# concatenating both acceptable and except_rued range
loan_df1 = pd.concat([loan_df1_include_ru,loan_df1_except_ru])
loan_df1.shape

#%%
loan_df1.boxplot(column= 'revol_util')
plt.show()
loan_df1['revol_util'].hist(bins=20)
loan_df1['revol_util'].describe()
#####################################
#%%
loan_df1.boxplot()
plt.xticks(rotation=90) 
plt.show()


#%%
# Sorting the data according to objective
#%%
test_list=["Jun-2015","Jul-2015","Aug-2015","Sep-2015","Oct-2015","Nov-2015","Dec-2015"]
test_list
#%%
test = loan_df1.loc[loan_df1.issue_d.isin(test_list)]
train = loan_df1.loc[-loan_df1.issue_d.isin(test_list)]

#%%

from sklearn import preprocessing
le={}                                          # create blank dictionary
for x in colname1:
    le[x]=preprocessing.LabelEncoder()  
        
for x in colname1:
    test[x]= le[x].fit_transform(test.__getattr__(x))

for x in colname1:
    train[x]= le[x].fit_transform(train.__getattr__(x)) 

#%%
X_train = train.values[:,:-1]
Y_train= train.values[:,-1]
Y_train= Y_train.astype(int)

X_test = test.values[:,:-1]

#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train,X_test)
x=scaler.transform(X_train,X_test)
print(X_train)


#%%
# Training a Logistic Regression Model.
from sklearn.linear_model import LogisticRegression
# create a model
classifier = (LogisticRegression())

# fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(list(Y_pred))
Y_pred_LR = list(Y_pred)

#%%
acc_classifier = round(classifier.score(X_train, Y_train) * 100, 2)
acc_classifier
#%%
# Using cross Validation

classifier=(LogisticRegression())
from sklearn import cross_validation

# Performing kfold cross validation

kfold_cv = cross_validation.KFold(n=len(X_train),n_folds = 10)
print(kfold_cv)

# running the model using scoring metric accuracy

kfold_cv_result = cross_validation.cross_val_score( \
                                        estimator =classifier,
                                        X=X_train,y=Y_train,
                                        scoring="accuracy",
                                        cv=kfold_cv)
print(kfold_cv_result)
# finding the mean
print(kfold_cv_result.mean())
#%%
#Running Decision Tree Model
#predicting using the decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier()
#fit the model on data and predict the values
model_DecisionTree.fit(X_train,Y_train)
Y_pred=model_DecisionTree.predict(X_test)

#%%
acc_model_DecisionTree = round(model_DecisionTree.score(X_train, Y_train) * 100, 2)
acc_model_DecisionTree

#%%
#using cross validation for Decision Tree
classifier=(DecisionTreeClassifier())
from sklearn import cross_validation
#performing kfold cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,
                                                 scoring='accuracy',cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#%%
#Running Extra Tree Classifier model for improving the performance of model
#Predicting using Bagging classifier
from sklearn.ensemble import ExtraTreesClassifier
model_ExtraTreesClassifier=(ExtraTreesClassifier(21))
#fit the model on data and predict the values
model_ExtraTreesClassifier.fit(X_train,Y_train)
Y_pred=model_ExtraTreesClassifier.predict(X_test)    

#%%
#Running Random Forest Model
#Predicting using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForestClassifier=(RandomForestClassifier(501))

#fit the model on data and predict the values
model_RandomForestClassifier.fit(X_train,Y_train)
Y_pred=model_RandomForestClassifier.predict(X_test)  

#%%
#predicting using AdaBoast Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoast=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))

#fit the model on data and predict the values
model_AdaBoast.fit(X_train,Y_train)
Y_pred=model_AdaBoast.predict(X_test)

#%%
#Running Gradient Boosting Classifier
#predicting using Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoostingClassifier=(GradientBoostingClassifier())

#fit the model on data and predict the values
model_GradientBoostingClassifier.fit(X_train,Y_train)
Y_pred=model_GradientBoostingClassifier.predict(X_test)

#%%
#Ensemble modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Crate Sub models
estimators = []
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
#%%
#create the ensemble model
ensemble=VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)

#%%
#using cross validation for Ensembeled model
classifier=VotingClassifier(estimators)
from sklearn import cross_validation
#performing kfold cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,
                                                 scoring='accuracy',cv=kfold_cv)
print(kfold_cv_result)

#finding the mean
print(kfold_cv_result.mean())

# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:24:34 2018

@author: Chexki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%  Data Loading

loan_df= pd.read_csv(r'D:\DATA SCIENCE\Python\Assignment 1\XYZCorp_LendingData.txt',header=0,delimiter='\t',low_memory=False)
print(loan_df)

#%%
# Generating a dummy dataset

loan_df1 = pd.DataFrame.copy(loan_df)
print(loan_df1.describe(include= 'all'))

#%% Feature Selection

# Some of the variables are not helpful in order to build a predictive model, hence dropping.

loan_df1.drop(['id','member_id','funded_amnt_inv','grade','emp_title','pymnt_plan','desc','title','addr_state',
            'inq_last_6mths','mths_since_last_record','initial_list_status','mths_since_last_major_derog','policy_code','application_type'
            ,'annual_inc_joint','dti_joint','verification_status_joint','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m'
            ,'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
            'max_bal_bc','all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1,inplace=True)

print(loan_df1.head())

#%%
# Checking if missing values are present.
loan_df1.isnull().sum()

#%%
print(loan_df1.dtypes)

#%%
# Imputing categorical missing data with mode value

colname1=['term','sub_grade','emp_length','home_ownership','verification_status',
          'issue_d','purpose','zip_code','earliest_cr_line','last_pymnt_d',
          'next_pymnt_d','last_credit_pull_d']
for x in colname1[:]:
    loan_df1[x].fillna(loan_df1[x].mode()[0],inplace=True)
    
loan_df1.isnull().sum()

#%%
colname2=['mths_since_last_delinq','revol_util','collections_12_mths_ex_med',
          'total_rev_hi_lim']
for x in colname2[:]:
    loan_df1[x].fillna(loan_df1[x].mean(),inplace=True)
    
loan_df1.isnull().sum()

#%%
print(loan_df1.shape)
loan_df1.describe()    

########################################################################################
#%%         OUTLIERS
#%%   Exploratory analysis : Graphical representation to know more about data,
#     Using boxplot.

loan_df1.boxplot()
plt.xticks(rotation=90) 
plt.show()

#%%

# Handling Extreme Outliers by replacing the with mean of respective features. 

#%%
loan_df1.boxplot(column= 'dti')
plt.show()
# Histogram
loan_df1['dti'].hist(bins=20)
loan_df1['dti'].describe()
#%%
q1_dti = loan_df1['dti'].quantile(0.25)
q3_dti = loan_df1['dti'].quantile(0.75)
iqr_dti = q3_dti-q1_dti                                    # Interquantile range
low_dti = q1_dti-1.5*iqr_dti                               # Acceptable range
high_dti = q3_dti+1.5*iqr_dti

# meeting the acceptable range

loan_df1_include_dti = loan_df1.loc[(loan_df1['dti']> low_dti) & \
                                     (loan_df1['dti']< high_dti)]
# not meeting the acceptable range
loan_df1_except_dti = loan_df1.loc[(loan_df1['dti']<= low_dti) | \
                                    (loan_df1['dti']>= high_dti)]

print(loan_df1_include_dti.shape)   # no. of acceptable observations
print(loan_df1_except_dti.shape)    # no. of outliers
print(high_dti,low_dti)

#finding mean of the acceptable range
dti_mean = loan_df1_include_dti.dti.mean() 
print(dti_mean)

#imputing outliers values with mean value
loan_df1_except_dti.dti = dti_mean
# concatenating both acceptable and except_dtied range
loan_df1 = pd.concat([loan_df1_include_dti,loan_df1_except_dti])
loan_df1.shape

#%%
loan_df1.boxplot(column= 'dti')
plt.show()
loan_df1['dti'].hist(bins=20)
loan_df1['dti'].describe()
########################################################################################
#%%
loan_df1.boxplot(column= 'revol_util')
plt.show()
# Histogram
loan_df1['revol_util'].hist(bins=20)
loan_df1['revol_util'].describe()
#%%
# treating outliers
q1_ru = loan_df1['revol_util'].quantile(0.25)
q3_ru = loan_df1['revol_util'].quantile(0.75)
iqr_ru = q3_ru-q1_ru                                    # Interquantile range
low_ru = q1_ru-1.5*iqr_ru                               # Acceptable range
high_ru = q3_ru+1.5*iqr_ru

# meeting the acceptable range
loan_df1_include_ru = loan_df1.loc[(loan_df1['revol_util']> low_ru) & \
                                     (loan_df1['revol_util']< high_ru)]
# not meeting the acceptable range
loan_df1_except_ru = loan_df1.loc[(loan_df1['revol_util']<= low_ru) | \
                                    (loan_df1['revol_util']>= high_ru)]

print(loan_df1_include_ru.shape)   # no. of acceptable observations
print(loan_df1_except_ru.shape)    # no. of outliers
print(high_ru,low_ru)

#finding mean of the acceptable range
revol_util_mean = loan_df1_include_ru.revol_util.mean() 
print(revol_util_mean)

#imputing outliers values with mean value
loan_df1_except_ru.revol_util = revol_util_mean

# concatenating both acceptable and except_rued range
loan_df1 = pd.concat([loan_df1_include_ru,loan_df1_except_ru])
loan_df1.shape

#%%
loan_df1.boxplot(column= 'revol_util')
plt.show()
loan_df1['revol_util'].hist(bins=20)
loan_df1['revol_util'].describe()
#####################################
#%%
loan_df1.boxplot()
plt.xticks(rotation=90) 
plt.show()


#%%
# Sorting the data according to objective
#%%
test_list=["Jun-2015","Jul-2015","Aug-2015","Sep-2015","Oct-2015","Nov-2015","Dec-2015"]
test_list
#%%
test = loan_df1.loc[loan_df1.issue_d.isin(test_list)]
train = loan_df1.loc[-loan_df1.issue_d.isin(test_list)]

#%%

from sklearn import preprocessing
le={}                                          # create blank dictionary
for x in colname1:
    le[x]=preprocessing.LabelEncoder()  
        
for x in colname1:
    test[x]= le[x].fit_transform(test.__getattr__(x))

for x in colname1:
    train[x]= le[x].fit_transform(train.__getattr__(x)) 

#%%
X_train = train.values[:,:-1]
Y_train= train.values[:,-1]
Y_train= Y_train.astype(int)

X_test = test.values[:,:-1]

#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train,X_test)
x=scaler.transform(X_train,X_test)
print(X_train)


#%%
# Training a Logistic Regression Model.
from sklearn.linear_model import LogisticRegression
# create a model
classifier = (LogisticRegression())

# fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(list(Y_pred))
Y_pred_LR = list(Y_pred)

#%%
acc_classifier = round(classifier.score(X_train, Y_train) * 100, 2)
acc_classifier
#%%
# Using cross Validation

classifier=(LogisticRegression())
from sklearn import cross_validation

# Performing kfold cross validation

kfold_cv = cross_validation.KFold(n=len(X_train),n_folds = 10)
print(kfold_cv)

# running the model using scoring metric accuracy

kfold_cv_result = cross_validation.cross_val_score( \
                                        estimator =classifier,
                                        X=X_train,y=Y_train,
                                        scoring="accuracy",
                                        cv=kfold_cv)
print(kfold_cv_result)
# finding the mean
print(kfold_cv_result.mean())
#%%
#Running Decision Tree Model
#predicting using the decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_DecisionTree=DecisionTreeClassifier()
#fit the model on data and predict the values
model_DecisionTree.fit(X_train,Y_train)
Y_pred=model_DecisionTree.predict(X_test)

#%%
acc_model_DecisionTree = round(model_DecisionTree.score(X_train, Y_train) * 100, 2)
acc_model_DecisionTree

#%%
#using cross validation for Decision Tree
classifier=(DecisionTreeClassifier())
from sklearn import cross_validation
#performing kfold cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,
                                                 scoring='accuracy',cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#%%
#Running Extra Tree Classifier model for improving the performance of model
#Predicting using Bagging classifier
from sklearn.ensemble import ExtraTreesClassifier
model_ExtraTreesClassifier=(ExtraTreesClassifier(21))
#fit the model on data and predict the values
model_ExtraTreesClassifier.fit(X_train,Y_train)
Y_pred=model_ExtraTreesClassifier.predict(X_test)    

#%%
#Running Random Forest Model
#Predicting using Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForestClassifier=(RandomForestClassifier(501))

#fit the model on data and predict the values
model_RandomForestClassifier.fit(X_train,Y_train)
Y_pred=model_RandomForestClassifier.predict(X_test)  

#%%
#predicting using AdaBoast Classifier
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoast=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))

#fit the model on data and predict the values
model_AdaBoast.fit(X_train,Y_train)
Y_pred=model_AdaBoast.predict(X_test)

#%%
#Running Gradient Boosting Classifier
#predicting using Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoostingClassifier=(GradientBoostingClassifier())

#fit the model on data and predict the values
model_GradientBoostingClassifier.fit(X_train,Y_train)
Y_pred=model_GradientBoostingClassifier.predict(X_test)

#%%
#Ensemble modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Crate Sub models
estimators = []
model1=LogisticRegression()
estimators.append(('log',model1))
model2=DecisionTreeClassifier()
estimators.append(('cart',model2))
model3=SVC()
estimators.append(('svm',model3))
#%%
#create the ensemble model
ensemble=VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)

#%%
#using cross validation for Ensembeled model
classifier=VotingClassifier(estimators)
from sklearn import cross_validation
#performing kfold cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=Y_train,
                                                 scoring='accuracy',cv=kfold_cv)
print(kfold_cv_result)

#finding the mean
print(kfold_cv_result.mean())