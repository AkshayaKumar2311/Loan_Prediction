#Importing Libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

#Importing DataSet
trainData = pd.read_csv("TrainLoanData.csv")

##Performig EDA
trainData.shape
trainData.columns
myColumnNames = list(trainData.columns.values.tolist())
trainData.info()
trainData.isnull().sum()
summary = trainData.describe()

#Missing Value and Inconsistencies
trainData['Gender'].value_counts()
trainData['Gender'] = np.where(trainData['Gender'].isnull(), trainData['Gender'].mode() , trainData['Gender'])
trainData['Married'].value_counts()
trainData['Married'] = np.where(trainData['Married'].isnull(), trainData['Married'].mode() , trainData['Married'])
trainData['Dependents'].value_counts()

trainData['Dependents'] = np.where(trainData['Dependents'].isnull(), trainData['Dependents'].mode() , trainData['Dependents'])
trainData['Self_Employed'].value_counts()
trainData['Self_Employed'] = np.where(trainData['Self_Employed'].isnull(), trainData['Self_Employed'].mode() , trainData['Self_Employed'])
(trainData['LoanAmount']).mean() #146.412
(trainData['LoanAmount'].median()) #128
trainData['LoanAmount'] = np.where(trainData['LoanAmount'].isnull(),trainData['LoanAmount'].median(),trainData['LoanAmount'])
np.mean(trainData['Loan_Amount_Term']) #342.0
(trainData['Loan_Amount_Term'].value_counts()) #360.0
trainData['Loan_Amount_Term'] = np.where(trainData['Loan_Amount_Term'].isnull(),trainData['Loan_Amount_Term'].mode(),trainData['Loan_Amount_Term'])
trainData['Credit_History'].value_counts()
np.mean(trainData['Credit_History'])
(trainData['Credit_History'].median())
trainData['Credit_History'] = np.where(trainData['Credit_History'].isnull(),trainData['Credit_History'].mode(),trainData['Credit_History'])
trainData.isnull().sum()
backUp = trainData

trainData=backUp

trainData.describe()

trainData.columns

#list of float columns: ApplicantIncome,CoapplicantIncome,LoanAmount

sns.boxplot(trainData['ApplicantIncome'])#outlier found
sns.distplot(trainData['ApplicantIncome'])#right skewed
trainData['ApplicantIncome'].skew()#6.539513 --> skewness is high; we need to reduce skewness by transformation

trainData['ApplicantIncome'].value_counts()
trainData['ApplicantIncome'].isnull().sum()
trainData['ApplicantIncome'].count
#applicantIncome has skewed data and skewness value is high.Hence do the data transformation
#data transformation
trainData['ApplicantIncome_sq']=trainData['ApplicantIncome'].apply(lambda x:x*x)
trainData['ApplicantIncome_cb']=trainData['ApplicantIncome'].apply(lambda x:x*x*x)
trainData['ApplicantIncome_sqr']=np.sqrt(trainData['ApplicantIncome'])
trainData['ApplicantIncome_cbr']=np.cbrt(trainData['ApplicantIncome'])
trainData['ApplicantIncome_log']=np.log(trainData['ApplicantIncome'])#print((-1)*np.log(-8))#how to log for negative numbers
trainData['ApplicantIncome_sqr'].skew()#2.967
trainData['ApplicantIncome_cbr'].skew()#2.08
trainData['ApplicantIncome_log'].skew()#0.4795
trainData['ApplicantIncome_sq'].skew()#13
trainData['ApplicantIncome_cb'].skew()#18

from scipy import stats
trainData['ApplicantIncome_Zscore']=stats.zscore(trainData['ApplicantIncome'])
trainData['ApplicantIncome_Zscore'].skew()#6.53
#hence we are going to consider the log column for calculating outlier since the value is less
#function to calculate outlier

def detect_outliers(x):
    sorted(x)
    q1,q3=np.percentile(x,[25,75])
    iqr=q3-q1
    lower_bound=q1-(1.5*iqr)
    upper_bound=q3+(1.5*iqr)   
    return q1,q3,lower_bound,upper_bound
#call detect_outlier for ApplicantIncome_log
outlier_applicant_inc_list = detect_outliers(trainData['ApplicantIncome_log'])
check_applicant_inc_lower = outlier_applicant_inc_list[2] > trainData['ApplicantIncome_log']#check the number of values less than lower bound
check_applicant_inc_upper = outlier_applicant_inc_list[3] < trainData['ApplicantIncome_log']#check the number of values greater than upper bound
check_applicant_inc_lower.value_counts()  # 6 values
check_applicant_inc_upper.value_counts()  # 21 Values are greater than upper bound value
trainData['ApplicantIncome_log'].skew()     # 0.48
sns.boxplot(trainData['ApplicantIncome_log'])

trainData['ApplicantIncome_log_tranformed'] = np.where(outlier_applicant_inc_list[3]<trainData['ApplicantIncome_log'],outlier_applicant_inc_list[1],trainData['ApplicantIncome_log'])
sns.boxplot(trainData['ApplicantIncome_log_tranformed'])
trainData['ApplicantIncome_log_tranformed']=np.where(outlier_applicant_inc_list[2]>trainData['ApplicantIncome_log_tranformed'],outlier_applicant_inc_list[0],trainData['ApplicantIncome_log_tranformed'])
sns.boxplot(trainData['ApplicantIncome_log_tranformed'])
            # or - another type of transformation - start'''
trainData['ApplicantIncome_log_tranformed']=np.where(trainData['ApplicantIncome_log']< outlier_applicant_inc_list[2],outlier_applicant_inc_list[0],trainData['ApplicantIncome_log'])
sns.boxplot(trainData['ApplicantIncome_log_tranformed'])
trainData['ApplicantIncome_log_tranformed']=np.where(trainData['ApplicantIncome_log_tranformed']> outlier_applicant_inc_list[3],outlier_applicant_inc_list[1],trainData['ApplicantIncome_log_tranformed'])
                 #or - another type of transformation - end'''
sns.distplot(trainData['ApplicantIncome_log_tranformed'])

trainData['ApplicantIncome_log_tranformed'].describe()
trainData['ApplicantIncome_log_tranformed'].value_counts()
trainData['ApplicantIncome_log_tranformed'].skew()
'''
#Skewness Check : 

trainData['ApplicantIncome'].skew() #6.54
trainData['CoapplicantIncome'].skew() #7.5
trainData['LoanAmount'].skew() #2.74

#Before Transormation
sns.boxplot(trainData['ApplicantIncome'])
#Taking Cuve root
trainData['ApplicantIncome'] = trainData['ApplicantIncome'].apply(lambda x: (-1)*np.power(-x,1./3) if x<0 else np.power(x,1./3))
trainData['ApplicantIncome'].skew() #2.08
trainData['ApplicantIncome_log'] = trainData['ApplicantIncome'].apply(lambda x: (-1)*np.log(x) if x<0 else np.log(x))
trainData['ApplicantIncome_log'].skew() #0.48
sns.boxplot(trainData['ApplicantIncome_log'])
sns.distplot(trainData['ApplicantIncome_log'])
sns.distplot(trainData['ApplicantIncome'])

def detect_outliers(x):
sorted(x)
q1, q3= np.percentile(x,[25,75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr) 
upper_bound = q3 + (1.5 * iqr) 
return q1,q3,lower_bound,upper_bound

#annual_inc Column
outlier_applicant_inc = detect_outliers(trainData['ApplicantIncome_log'])
check_applicant_inc_lower = outlier_applicant_inc[2] > trainData['ApplicantIncome_log']
check_applicant_inc_upper = outlier_applicant_inc[3] < trainData['ApplicantIncome_log']
check_applicant_inc_lower.value_counts()  # 6 values
check_applicant_inc_upper.value_counts()  # 21 Values are greater than upper bound value
trainData['ApplicantIncome_log'].skew()     # 0.48

trainData['ApplicantIncome_log_transform']=np.where(outlier_applicant_inc[3] < trainData['ApplicantIncome_log'] ,outlier_applicant_inc[1],trainData['ApplicantIncome_log'])

trainData['ApplicantIncome_log_transform']=np.where(outlier_applicant_inc[2] > trainData['ApplicantIncome_log_transform'] ,outlier_applicant_inc[0],trainData['ApplicantIncome_log_transform'])
sns.boxplot(trainData['ApplicantIncome_log_transform'])

trainData['ApplicantIncome_log'][613]
'''

#skewness and outlier check for column: CoapplicantIncome
trainData.columns
trainData['CoapplicantIncome'].value_counts()#it has zero for 273 rows; need to impute the data using mean/median
trainData['ImputedCoapplicantIncome']=np.where(trainData['CoapplicantIncome']==0.0,trainData['CoapplicantIncome'].median(),trainData['CoapplicantIncome'])
trainData['ImputedCoapplicantIncome'].value_counts()
#to check the skewness
trainData['ImputedCoapplicantIncome'].skew()
sns.distplot(trainData['ImputedCoapplicantIncome'])#right skewed one
sns.boxplot(trainData['ImputedCoapplicantIncome'])#outlier found; only upper outlier

#data transformation
trainData['CoapplicantIncome_sq']=trainData['ImputedCoapplicantIncome'].apply(lambda x:x*x)
trainData['CoapplicantIncome_cb']=trainData['ImputedCoapplicantIncome'].apply(lambda x:x*x*x)
trainData['CoapplicantIncome_sqr']=np.sqrt(trainData['ImputedCoapplicantIncome'])
trainData['CoapplicantIncome_cbr']=np.cbrt(trainData['ImputedCoapplicantIncome'])
trainData['CoapplicantIncome_log']=np.log(trainData['ImputedCoapplicantIncome'])#print((-1)*np.log(-8))#how to log for negative numbers
trainData['CoapplicantIncome_sqr'].skew()#3.903
trainData['CoapplicantIncome_cbr'].skew()#2.666
trainData['CoapplicantIncome_log'].skew()#0.364
trainData['CoapplicantIncome_sq'].skew()#16
trainData['CoapplicantIncome_cb'].skew()#19
from scipy import stats
trainData['CoapplicantIncome_Zscore']=stats.zscore(trainData['ImputedCoapplicantIncome'])
trainData['CoapplicantIncome_Zscore'].skew()#9.22

#CoapplicantIncome_log has less skewed value; so CoapplicantIncome_log will be consider for outlier 
#callfunction detect_outliers to get the q1,q3,lowerbound and upperbound
sns.boxplot(trainData['CoapplicantIncome_log'])#noticed left and right outlier
trainData['CoapplicantIncome_log'].value_counts()
outlier_coapplicant_inc_list=detect_outliers(trainData['CoapplicantIncome_log'])

check_coapplication_upper=outlier_coapplicant_inc_list[3]<trainData['CoapplicantIncome_log']
check_coapplication_upper.value_counts()#18 rows has value greater than upper bound; need to replace with q3
check_coapplication_lower=outlier_coapplicant_inc_list[2]>trainData['CoapplicantIncome_log']
check_coapplication_lower.value_counts()#4 true: which means 4 value below lower bound; so, need to replace those 4 values with q1

#update the values greater than upper bound values to q3
trainData['TransformedCoapplicantIncome']=np.where(trainData['CoapplicantIncome_log']>outlier_coapplicant_inc_list[3],outlier_coapplicant_inc_list[1],trainData['CoapplicantIncome_log'])
trainData['TransformedCoapplicantIncome']=np.where(trainData['TransformedCoapplicantIncome']<outlier_coapplicant_inc_list[2],outlier_coapplicant_inc_list[0],trainData['TransformedCoapplicantIncome'])
sns.boxplot( trainData['TransformedCoapplicantIncome'])
sns.distplot(trainData['TransformedCoapplicantIncome'])

trainData['TransformedCoapplicantIncome'].value_counts()
trainData['TransformedCoapplicantIncome'].skew()#0.7868 - skewness increases after outlier. need to check why


#skewness and outlier check for column: LoanAmount
trainData.columns
#for loanAmount, already missing values are imputed in row#37; no 0 in the entire column; validated using value_counts
trainData['LoanAmount'].value_counts()
trainData['LoanAmount']
sns.distplot(trainData['LoanAmount']) #right skewed
sns.boxplot(trainData['LoanAmount']) # upper outlier found

#data transformation
trainData['LoanAmount_sq']=trainData['LoanAmount'].apply(lambda x:x*x)
trainData['LoanAmount_cb']=trainData['LoanAmount'].apply(lambda x:x*x*x)
trainData['LoanAmount_sqr']=np.sqrt(trainData['LoanAmount'])
trainData['LoanAmount_cbr']=np.cbrt(trainData['LoanAmount'])
trainData['LoanAmount_log']=np.log(trainData['LoanAmount'])#print((-1)*np.log(-8))#how to log for negative numbers
trainData['LoanAmount_sqr'].skew()#1.34
trainData['LoanAmount_cbr'].skew()#.86
trainData['LoanAmount_log'].skew()# -.19
trainData['LoanAmount_sq'].skew()#5.48
trainData['LoanAmount_cb'].skew()#7.87
from scipy import stats
trainData['LoanAmount_Zscore']=stats.zscore(trainData['LoanAmount'])
trainData['LoanAmount_Zscore'].skew()#2.74

#LoanAmount_log will be considered and outlier will be calculated
sns.boxplot(trainData['LoanAmount_log'])
sns.distplot(trainData['LoanAmount_log'])

#replace the upper outlier data with q3 and lower outlier with q1
#call detect_outliers to calculate q1,q3,lowerbound and upperbound

loan_amount_outlier_list=detect_outliers(trainData['LoanAmount_log'])
print(f"value of Q1  is  {loan_amount_outlier_list[0]}" + f"value of q3 is {loan_amount_outlier_list[1]}")

check_loan_amount_lower=loan_amount_outlier_list[2]>trainData['LoanAmount_log']#check the number of values less than lower outlier
check_loan_amount_lower.value_counts()# 18 true --> 18 value less than lower outlier

check_loan_amount_upper=loan_amount_outlier_list[3]<trainData['LoanAmount_log']#check the number of values greater than upper outlier
check_loan_amount_upper.value_counts()# 21 true --> 21 value greater than upper outlier

#data transformation for loan amount

#only one tranformation will be enough i.e replace the upper outlier value with q3; lower outlier not available
trainData['transformedLoanAmount']=np.where(trainData['LoanAmount_log']>loan_amount_outlier_list[3],loan_amount_outlier_list[1],trainData['LoanAmount_log'])
trainData['transformedLoanAmount']=np.where(trainData['transformedLoanAmount']<loan_amount_outlier_list[2],loan_amount_outlier_list[0],trainData['transformedLoanAmount'])

sns.boxplot(trainData['transformedLoanAmount'])
sns.distplot(trainData['transformedLoanAmount'])

trainData['transformedLoanAmount'].skew() #003308
trainData['transformedLoanAmount'].value_counts()


#####Lable Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
trainData['Gender'] = le.fit_transform(trainData['Gender'])
trainData['Married'] = le.fit_transform(trainData['Married'])
trainData['Dependents'] = le.fit_transform(trainData['Dependents'])
trainData['Education'] = le.fit_transform(trainData['Education'])
trainData['Self_Employed'] = le.fit_transform(trainData['Self_Employed'])
trainData['Property_Area'] = le.fit_transform(trainData['Property_Area'])
trainData['Loan_Status'] = le.fit_transform(trainData['Loan_Status'])

# =============================================================================
# forMyTransformation = ['Gender','Married','Dependents','Education','Self_Employed',\
#'Property_Area','Loan_Status']
# 
# for i in forMyTransformation:
#     print(le.fit_transform(i))
# =============================================================================

#Assigning X and Y

X = trainData[['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Amount_Term' ,\
               'ApplicantIncome_log_tranformed','TransformedCoapplicantIncome','transformedLoanAmount']]

Y = trainData['Loan_Status']

#TestTrainSplit
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y= train_test_split(X,Y, test_size=0.2)


from sklearn.linear_model import LogisticRegression
Logistic_Reg_train = LogisticRegression()
Logistic_Reg_train.fit(train_x,train_y)
predict_logistics_y = Logistic_Reg_train.predict(train_x)

from sklearn.metrics import confusion_matrix
cm_logisticR = confusion_matrix(train_y,predict_logistics_y)

from sklearn.svm import SVC
svc_rbf_train = SVC(kernel='rbf')
SupportVectorMachine_Model = svc_rbf_train.fit(train_x,train_y)
preds_svc_ytrain = SupportVectorMachine_Model.predict(train_x)
cm_svm_train_Y = confusion_matrix(train_y,preds_svc_ytrain)


from sklearn.ensemble import RandomForestClassifier
rf_train= RandomForestClassifier(n_estimators=100)
randomForest_Model = rf_train.fit(train_x,train_y)
predictModel_rf = randomForest_Model.predict(train_x)
cm_randomForestR = confusion_matrix(train_y,predictModel_rf)


from sklearn.svm import SVC
svc_rbf = SVC(kernel='poly')
SupportVectorMachine_Model = svc_rbf.fit(train_x,train_y)
preds_svc = SupportVectorMachine_Model.predict(train_x)
cm_svm = confusion_matrix(train_y,preds_svc)


######Clustering Methods##############


X_continuous = X[['Applicant_Income','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]

from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 2, random_state = 2)
KMeans.fit(X_continuous)
pred_clusters = KMeans.predict(X_continuous)
