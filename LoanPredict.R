#working director
setwd('CUsersChanduDesktopIMAR DATAloan prediction')
getwd()

#import train and test
train_dataset - read.csv('train_u6lujuX_CVtuZ9i.csv',header = TRUE,sep = ,)
test_dataset - read.csv('test_Y3wMUE5_7gLdaTN.csv', header = TRUE, sep = ,)

summary(train_dataset)
summary(test_dataset)
str(train_dataset)
str(test_dataset)
library(psych)
describe(train_dataset)

train_dataset$CoapplicantIncome - as.numeric(train_dataset$CoapplicantIncome)
train_dataset$ApplicantIncome - as.numeric(train_dataset$ApplicantIncome)
train_dataset$Loan_Amount_Term - as.numeric(train_dataset$Loan_Amount_Term)


#gender train
table(train_dataset$Gender)
train_dataset$Gender - as.character(train_dataset$Gender)
class(train_dataset$Gender)
train_dataset$Gender[train_dataset$Gender== '']- Male
train_dataset$Gender - as.factor(train_dataset$Gender)
#gender test
table(test_dataset$Gender)
test_dataset$Gender - as.character(test_dataset$Gender)
class(test_dataset$Gender)
test_dataset$Gender[test_dataset$Gender== '']- Male
test_dataset$Gender - as.factor(test_dataset$Gender)
#married train
table(train_dataset$Married)
train_dataset$Married - as.character(train_dataset$Married)
class(train_dataset$Married)
train_dataset$Married[train_dataset$Married== ''] - 'Yes'
train_dataset$Married- as.factor(train_dataset$Married)

#married trainsform

#train_dataset$Gender - as.character(train_dataset$Gender)
#train_dataset$Married- as.character(train_dataset$Married)
#train_dataset$Married - ifelse(train_dataset$Gender=='Male',
 #                                ifelse(train_dataset$CoapplicantIncome=='0',
  #                                      'No',train_dataset$Married),
   #                              train_dataset$Married)
#train_dataset$Gender - as.factor(train_dataset$Gender)
#train_dataset$Married- as.factor(train_dataset$Married)

#married transform female

train_dataset$Gender - as.character(train_dataset$Gender)
train_dataset$Married- as.character(train_dataset$Married)
train_dataset$Married - ifelse(train_dataset$Gender=='Female',
                                ifelse(train_dataset$CoapplicantIncome=='0',
                                       'No',train_dataset$Married),
                                train_dataset$Married)

train_dataset$Gender - as.factor(train_dataset$Gender)
train_dataset$Married- as.factor(train_dataset$Married)

#married transform test female

test_dataset$Gender - as.character(test_dataset$Gender)
test_dataset$Married- as.character(test_dataset$Married)
test_dataset$Married - ifelse(test_dataset$Gender=='Female',
                                ifelse(test_dataset$CoapplicantIncome=='0',
                                       'No',test_dataset$Married),
                                test_dataset$Married)

test_dataset$Gender - as.factor(test_dataset$Gender)
test_dataset$Married- as.factor(test_dataset$Married)

#dependency train
table(train_dataset$Dependents)
train_dataset$Dependents - as.character(train_dataset$Dependents)
class(train_dataset$Dependents)
train_dataset$Dependents[train_dataset$Dependents=='']- '0'
train_dataset$Dependents - as.factor(train_dataset$Dependents)
#dependcy test
table(test_dataset$Dependents)
test_dataset$Dependents - as.character(test_dataset$Dependents)
class(test_dataset$Dependents)
test_dataset$Dependents[test_dataset$Dependents=='']- '0'
test_dataset$Dependents - as.factor(test_dataset$Dependents)

#self employed train
table(train_dataset$Self_Employed)
train_dataset$Self_Employed - as.character(train_dataset$Self_Employed)
class(train_dataset$Self_Employed)
train_dataset$Self_Employed[train_dataset$Self_Employed=='']- 'No'
train_dataset$Self_Employed - as.factor(train_dataset$Self_Employed)
#seelf employed test
table(test_dataset$Self_Employed)
test_dataset$Self_Employed - as.character(test_dataset$Self_Employed)
class(test_dataset$Self_Employed)
test_dataset$Self_Employed[test_dataset$Self_Employed=='']- 'No'
test_dataset$Self_Employed - as.factor(test_dataset$Self_Employed)

#loan amount train
sum(is.na(train_dataset$LoanAmount))
train_dataset$LoanAmount - ifelse(is.na(train_dataset$LoanAmount),
                                          median(train_dataset$LoanAmount,na.rm = TRUE),
                                          train_dataset$LoanAmount)
#loan amount test
sum(is.na(test_dataset$LoanAmount))
test_dataset$LoanAmount - ifelse(is.na(test_dataset$LoanAmount),
                                   median(test_dataset$LoanAmount,na.rm = TRUE),
                                   test_dataset$LoanAmount)

#using sub
#train_dataset - sub(^$,Male,train)_datset$gender

#loan amount term train
sum(is.na(train_dataset$Loan_Amount_Term))
train_dataset$Loan_Amount_Term - ifelse(is.na(train_dataset$Loan_Amount_Term),
                                                median(is.na(train_dataset$Loan_Amount_Term),na.rm= TRUE),
                                                 train_dataset$Loan_Amount_Term)
#loan amount test
sum(is.na(test_dataset$Loan_Amount_Term))
test_dataset$Loan_Amount_Term - ifelse(is.na(test_dataset$Loan_Amount_Term),
                                         median(is.na(test_dataset$Loan_Amount_Term),na.rm= TRUE),
                                         test_dataset$Loan_Amount_Term)

#credit history train
table(train_dataset$Credit_History)
sum(is.na(train_dataset$Credit_History))
train_dataset$Credit_History - ifelse(is.na(train_dataset$Credit_History),
                                        '1',
                                        train_dataset$Credit_History)
train_dataset$Credit_History- as.factor(train_dataset$Credit_History)

#credit history test
sum(is.na(test_dataset$Credit_History))
test_dataset$Credit_History - ifelse(is.na(test_dataset$Credit_History),
                                       '1',
                                       test_dataset$Credit_History)
test_dataset$Credit_History- as.factor(test_dataset$Credit_History)

#model
model_1 - glm(Loan_Status ~ Gender + Married + Dependents + Education +      
                      Self_Employed+ ApplicantIncome+ CoapplicantIncome+ LoanAmount
                      +  Loan_Amount_Term+ Credit_History +Property_Area,
                       family = binomial, data = train_dataset)

summary(model_1)
train_dataset$predicted_loanstatus - predict(model_1, train_dataset,type = 'response')
train_dataset$predicted_loanstatus - ifelse(train_dataset$predicted_loanstatus  0.5,
                                              'Y','N')
train_dataset$predicted_loanstatus- as.factor(train_dataset$predicted_loanstatus)

#confusing matrix
cm - table(train_dataset$Loan_Status, train_dataset$predicted_loanstatus)
cm
accuracy - ((84+415)614)100
accuracy

#decision tree
library(party)
png(file=loandecisiontree.png)
set.seed(123)#random trees generations
str(train_dataset)
decision_tree - ctree(Loan_Status ~ Gender + Married + Dependents + Education +      
                         Self_Employed+ ApplicantIncome+ CoapplicantIncome+ LoanAmount
                        + Loan_Amount_Term+ Credit_History +Property_Area,
                          data = train_dataset)
summary(decision_tree)
decision_tree
plot(decision_tree)
dev.off()
train_dataset$predict_decison_tree - predict(decision_tree,train_dataset,type= 'response')
#confusing matrix decison tree
cm_1 - table(train_dataset$Loan_Status,train_dataset$predict_decison_tree)
cm_1
accuracy_decision_tree -((82+415)614)100
accuracy_decision_tree
str(train_dataset)
str(test_dataset)
#random forest
train_dataset$CoapplicantIncome - as.numeric(train_dataset$CoapplicantIncome)
library(randomForest)
model_random_forest - randomForest(Loan_Status ~ Gender + Married + Dependents + Education +      
                                      Self_Employed+ ApplicantIncome+ CoapplicantIncome+ LoanAmount
                                    + Loan_Amount_Term+ Credit_History +Property_Area,
                                      data = train_dataset)

train_dataset$predict_randomforest_loanstatus- predict(model_random_forest,train_dataset,
                                                        type = 'response')
#confusing matrix for random forest
cm_2 -table(train_dataset$Loan_Status,train_dataset$predict_randomforest_loanstatus)
cm_2
accuracy_random_forest - ((186+422)614)100
accuracy_random_forest


#support vector
#no of support vector shuld be less make as much less as possible
#install.packages('e1071')
library(e1071)
model_support_vector_radial - svm(Loan_Status ~ Gender + Married + Dependents + Education +      
                               Self_Employed+ ApplicantIncome+ CoapplicantIncome+ LoanAmount
                             + Loan_Amount_Term+ Credit_History +Property_Area,
                               data = train_dataset)
model_support_vector_radial
train_dataset$predicted_svm - predict(model_support_vector_radial,train_dataset)
cm_svm - table(train_dataset$Loan_Status, train_dataset$predicted_svm)
cm_svm
accuracy_svm - ((85+416)614)100
accuracy_svm

#polynomial kernel= polynomial, sigmoid, radial
model_support_vector_poly - svm(Loan_Status ~ Gender + Married + Dependents + Education +      
                                     Self_Employed+ ApplicantIncome+ CoapplicantIncome+ LoanAmount
                                   + Loan_Amount_Term+ Credit_History +Property_Area,
                                    kernel='polynomial',
                                   data = train_dataset)
model_support_vector_poly

train_dataset$predicted_poly - predict(model_support_vector_poly,train_dataset)
cm_svm_1 - table(train_dataset$Loan_Status, train_dataset$predicted_poly)
cm_svm_1
accuracy_svm1 <- ((13+422)614)100
accuracy_svm1

#caret package
#install.packages('caret')
#applying on test
test_dataset$Loan_status - predict(model_random_forest, test_dataset)
write.csv(test_dataset,sample_submission.csv)