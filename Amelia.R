#working director
setwd('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\loan prediction')
getwd()

#import train and test use na.strins='' it will impute as NA
train_dataset <- read.csv('train_u6lujuX_CVtuZ9i.csv',header = TRUE,sep = ",",na.strings = '')
test_dataset <- read.csv('test_Y3wMUE5_7gLdaTN.csv', header = TRUE, sep = ",",na.strings = '')

str(train_dataset)
str(test_dataset)
#train_dataset$Gender <- sub("^$",'NA',train_dataset$Gender)
sum(is.na(train_dataset$Gender))
sum(is.na(train_dataset$Married))
sum(is.na(train_dataset$Dependents))
sum(is.na(train_dataset$Self_Employed))
sum(is.na(train_dataset$LoanAmount))
sum(is.na(train_dataset$Loan_Amount_Term))
sum(is.na(train_dataset$Credit_History))

#amelia package
#install.packages('Amelia')
# idvars which are not to be imputed/continous
#noms categorical value impute
# ords for ordibnal variable
library(Amelia)
names(train_dataset)
impute_amelia <- amelia(train_dataset, m=5,
                         idvars = c("Loan_ID","Education","Property_Area","Loan_Status"),
                        noms =c("Gender","Married","Dependents","Self_Employed","Credit_History"))

 #new data frame after imputations
train_new_dataset <- as.data.frame(impute_amelia$imputations)