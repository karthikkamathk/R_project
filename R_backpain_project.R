

load("F:/UCD/Second Sem/Stat_Machine_learning/statMac_project/backpain.RData") #loading the workspace containing  the data and variable details.

backpain = dat
attach(backpain)

sum(is.na(backpain)) #checking the presence of missing values.
N = nrow(backpain)

str(backpain)

set.seed(18201357)

# Numerical categorical data
backpain$SurityRating = as.factor(backpain$SurityRating)
str(backpain)

# Libraries used

library(nnet)
library(caret)
library(rpart)
library(randomForest)
library(adabag)
library(kernlab)
library(glmnet)


num.iter = 100
results = matrix(0, num.iter, 8)
colnames(results) <- c("Classification Tree", "L.R using multinorm", "Random Forest", 
                       "SVM", "Bagging","Boosting", "Best Method", "Test Accuracy")

for (i in 1 : num.iter)
{
  #Splitting dataset into train data, validation data and test data.
  trainIndex <- sample(1:N, replace=TRUE) #Bootstrap the data for train
  trainIndex <- sort(trainIndex)
  restDataInd <- setdiff(1:N,trainIndex)
  
  validationIndex = sample(setdiff(1:N,trainIndex),size=0.50*(length(restDataInd)))
  testIndex = setdiff(1:N,union(trainIndex,validationIndex))
  
  #########################################################################################
  ###############################  Classification Trees  ##################################
  #########################################################################################
  
  fit.classfctn = rpart(PainDiagnosis ~ ., data = backpain, subset = trainIndex)
  pred.classfctn = predict(fit.classfctn, type = "class", newdata = backpain[validationIndex,])
  
  tab.classfctn = table(pred.classfctn, PainDiagnosis[validationIndex])
  acc.classfctn = sum(diag(tab.classfctn))/sum(tab.classfctn)
  
  results[i, 1] = acc.classfctn
  #########################################################################################
  ###############################  logistical Regression  #################################
  #########################################################################################
  
  #Logistic regression using multinorm function
  
  fit.logistic = multinom(PainDiagnosis ~ ., data = backpain, subset = trainIndex)
  pred.logistic = predict(fit.logistic, type = "class", newdata = backpain[validationIndex,])
  
  tab.logistic = table(pred.logistic, PainDiagnosis[validationIndex])
  acc.logistic = sum(diag(tab.logistic))/sum(tab.logistic)
  
  results[i, 2] = acc.logistic
  
  #########################################################################################
  ################################### Random Forest  ######################################
  #########################################################################################
  
  fit.randomf = randomForest(PainDiagnosis ~ ., data = backpain[trainIndex,])
  pred.randomf = predict(fit.randomf, type = "class", newdata = backpain[validationIndex, ])
  
  tab.randomf = table(pred.randomf, PainDiagnosis[validationIndex])
  acc.randomf = sum(diag(tab.randomf))/sum(tab.randomf)
  
  results[i, 3] = acc.randomf
  
  #########################################################################################
  ################################ Support Vector Machine #################################
  #########################################################################################
  
  fit.svm = ksvm(PainDiagnosis ~ ., data = backpain[trainIndex, ])
  pred.svm = predict(fit.svm, newdata = backpain[validationIndex, ])
  
  tab.svm = table(pred.svm, PainDiagnosis[validationIndex])
  acc.svm = sum(diag(tab.svm))/sum(tab.svm)
  
  results[i, 4]= acc.svm
  
  ######################################################################################
  #####################################   Bagging   ####################################
  ######################################################################################
  
  fit.bagg = bagging(PainDiagnosis ~ ., data = backpain[trainIndex,])
  pred.bagg = predict(fit.bagg, type = "class", newdata = backpain[validationIndex, ])
  acc.bagg = 1-pred.bagg$error
  
  results[i, 5]= acc.bagg
  
  ######################################################################################
  #####################################   Boosting   ###################################
  ######################################################################################
  
  fit.adaboost <- boosting(PainDiagnosis ~ ., data = backpain[trainIndex,])
  pred.adaboost <- predict.boosting(fit.adaboost,backpain[validationIndex, ])
  acc.adaboost = 1- pred.adaboost$error
  
  results[i, 6]= acc.adaboost
  
  # To check the best method among 6 classifiers
  max_accIndex = which.max(results[i , 1:6])
  
  
  # Finding the best method in each iteration.
  switch(max_accIndex,
         "1" = {
           results[i,7 ] = "Classification Tree"
           pred.test = predict(fit.classfctn,type="class",newdata=backpain[testIndex,])
           tab.test = table(pred.test , PainDiagnosis[testIndex])
           acc.test = sum(diag(tab.test))/sum(tab.test)
           results[i,8] = acc.test
         },
         "2" = {
           results[i,7] = "L.R using mutlinorm"
           pred.test = predict(fit.logistic,type="class",newdata=backpain[testIndex,])
           tab.test = table(pred.test , PainDiagnosis[testIndex])
           acc.test = sum(diag(tab.test))/sum(tab.test)
           results[i,8] = acc.test
         },
         "3" = {
           results[i,7] = "Random Forest"
           pred.test = predict(fit.randomf,type="class",newdata=backpain[testIndex,])
           tab.test = table(pred.test , PainDiagnosis[testIndex])
           acc.test = sum(diag(tab.test))/sum(tab.test)
           results[i,8] = acc.test
         },
         "4" = {
           results[i,7] = "SVM"
           pred.test = predict(fit.svm,newdata=backpain[testIndex,])
           tab.test = table(pred.test , PainDiagnosis[testIndex])
           acc.test = sum(diag(tab.test))/sum(tab.test)
           results[i,8] = acc.test
         },
         "5" = {
           results[i,7] = "Bagging"
           pred.test = predict(fit.bagg,type="class",newdata=backpain[testIndex,])
           acc.test = 1 - pred.test$error
           results[i,8] = acc.test
         },
         "6" = {
           results[i,7] = "Boosting"
           pred.test <- predict.boosting(fit.adaboost,backpain[testIndex,])
           acc.test = 1 - pred.test$error
           results[i,8] = acc.test})
}

head(results) #

resultSummary = as.data.frame(apply(results[,-7],2, as.numeric))
summary(resultSummary) #getting the summary statitics of 100 iterations

table(results[,7]) #getting the count of best methods for 100 iterations.


#The best model is found out to be Random Forest. We can find the performance 
#measures of this model by fitting it again with train data.

fit.randomf = randomForest(PainDiagnosis ~ ., data = backpain[trainIndex,])
pred.randomf = predict(fit.randomf, type = "class", newdata = backpain[testIndex, ])

library(caret) #Library used for finding confusion matrix and performance parameters
confusionMatrix(data=pred.randomf, reference = PainDiagnosis[testIndex])


# Variable importance plots can be produced for random forests
varImp(fit.randomf)    # Values
varImpPlot(fit.randomf) # Plot


############# End of Code ################