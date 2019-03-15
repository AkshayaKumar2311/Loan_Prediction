import pandas as pd
import numpy as np
import os
import math

LoantrainData = pd.read_csv('TrainLoanData.csv')

LoantestData = pd.read_csv('TestLoanData.csv')
LoantestData.dtypes
# =============================================================================
# 
# for k in LoantestData:
#     if(LoantestData.dtypes == object):
#             print("Hello!")
# =============================================================================
            
# =============================================================================
# for k in LoantestData:
#     np.isnan(LoantestData)
# =============================================================================


LoantrainData.columns.values
summary = LoantrainData.describe()
LoantrainData.info()

LoantrainData['Gender'].value_counts()

genderDummy = np.where(LoantrainData['Gender'].isnull(),'Male',LoantrainData['Gender'])
LoantrainData['Gender'] = genderDummy

LoantrainData['Married'].value_counts()
MarriedDummy = np.where(LoantrainData['Married'].isnull(),'Yes',LoantrainData['Married'])
LoantrainData['Married'] = MarriedDummy

LoantrainData['Dependents'].value_counts()
dependentsDummy = np.where(LoantrainData['Dependents'].isnull(),'0',LoantrainData['Dependents'])
LoantrainData['Dependents'] = dependentsDummy
LoantrainData['Dependents'].replace('3+','3',inplace= True)

LoantrainData['Education'].value_counts()


LoantrainData['Self_Employed'].value_counts()
employedDummy = np.where(LoantrainData['Self_Employed'].isnull(),'No',LoantrainData['Self_Employed'])
LoantrainData['Self_Employed'] = employedDummy

LoantrainData['LoanAmount'].median()
loanDummy = np.where(LoantrainData['LoanAmount'].isnull(),128,LoantrainData['LoanAmount'])
LoantrainData['LoanAmount'] = loanDummy

LoantrainData['Loan_Amount_Term'].median()
loantermDummy = np.where(LoantrainData['Loan_Amount_Term'].isnull(),360,LoantrainData['Loan_Amount_Term'])
LoantrainData['Loan_Amount_Term'] = loantermDummy


LoantrainData['Credit_History'].value_counts()
creditDummy = np.where(LoantrainData['Credit_History'].isnull(),1,LoantrainData['Credit_History'])
LoantrainData['Credit_History'] = creditDummy



LoantrainData.info()

#####Lable Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

LoantrainData['Gender']=le.fit_transform(LoantrainData['Gender'])
LoantrainData['Married']=le.fit_transform(LoantrainData['Married'])
LoantrainData['Dependents']=le.fit_transform(LoantrainData['Dependents'])
LoantrainData['Self_Employed']=le.fit_transform(LoantrainData['Self_Employed'])
LoantrainData['Education']=le.fit_transform(LoantrainData['Education'])

LoantrainData['Property_Area']=le.fit_transform(LoantrainData['Property_Area'])
LoantrainData['Loan_Status']=le.fit_transform(LoantrainData['Loan_Status'])


#############For Loop making work easy
# =============================================================================
# dummyFrTransform = ['Gender','Married','Dependents','Self_Employed','Education','Property_Area','Loan_Status']
# for i in dummyFrTransform:
#     LoantrainData[i] = le.fit_transform(LoantrainData[i])
#LoantrainData.dtypes

# =============================================================================

Z= LoantrainData.iloc[:,1:-1]
Y= LoantrainData['Loan_Status']
X= LoantrainData.iloc[:,2:12]

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,Y)
preds_lr = clf.predict(X)

from sklearn.metrics import confusion_matrix

cm_logisticR = confusion_matrix(Y,preds_lr)


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=500)
randomForest_Model = rf.fit(X,Y)
predictModel_rf = randomForest_Model.predict(X)
cm_randomForestR = confusion_matrix(Y,predictModel_rf)


from sklearn.svm import SVC
svc_rbf = SVC(kernel='rbf')
SupportVectorMachine_Model = svc_rbf.fit(X,Y)
preds_svc = SupportVectorMachine_Model.predict(X)
cm_svm = confusion_matrix(Y,preds_svc)




######Clustering Methods##############

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X,Y)

preds_lr = clf.predict(X)

from sklearn.metrics import confusion_matrix

cm_logisticR = confusion_matrix(Y,preds_lr)

X_continuous = X[['Applicant_Income','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]

from sklearn.cluster import KMeans

KMeans = KMeans(n_clusters = 2, random_state = 2)

KMeans.fit(X_continuous)

pred_clusters = KMeans.predict(X_continuous)
