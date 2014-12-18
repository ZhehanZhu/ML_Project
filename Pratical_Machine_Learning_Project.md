Predicting the way in which people perform barbell lifts
==============================================================

##Synopsis
In this report we aim to predict the way in which people perform barbell lifts correctly and incorrectly. To investigate this problem, I obtained data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. More information is available [here](http://groupware.les.inf.puc-rio.br/har).

##Model Building
###Feature Selection
Under the instruction of paper [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf), I decided to extract 12 features from [training data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) to make the prediction, including 
the maximum, range and variance of the belt accelerometer vector, variance of the belt gyro and magnetometer, the variance of the arm accelerometer vector and the maximum and minimum of the arm magnetometer, the maximum of the dumbbell acceleration, the maximum and minimum of the dumbbell magnetometer,and the maximum and minimum of the glove gyro.


```r
trainData.ori <- read.table("pml-training.csv", header = TRUE, sep = ",", na.strings = c("","NA"))
col.ind <- which(apply(trainData.ori[,], 2, FUN = function(x) sum(complete.cases(x))) == 19622)
trainData <- trainData.ori[, col.ind[-1]]


max_belt_accel <- apply(X = cbind(trainData$accel_belt_x,trainData$accel_belt_y,trainData$accel_belt_z),1,
                        FUN = function(x) max(x))
range_belt_accel <- apply(X = cbind(trainData$accel_belt_x,trainData$accel_belt_y,trainData$accel_belt_z),1,
                   FUN = function(x) max(x)-min(x))
var_belt_accel <- apply(X = cbind(trainData$accel_belt_x,trainData$accel_belt_y,trainData$accel_belt_z),1,
                  FUN = function(x) var(x))
var_belt_mag <- apply(X = cbind(trainData$magnetl_belt_x,trainData$magnet_belt_y,trainData$magnet_belt_z),1,
                      FUN = function(x) var(x))
var_arm_accel <- apply(X = cbind(trainData$accel_arm_x,trainData$accel_arm_y,trainData$accel_arm_z),1,
                       FUN = function(x) var(x))
max_arm_mag <- apply(X = cbind(trainData$magnet_arm_x,trainData$magnet_arm_y,trainData$magnet_arm_z),1,
                     FUN = function(x) max(x))
min_arm_mag <- apply(X = cbind(trainData$magnet_arm_x,trainData$magnet_arm_y,trainData$magnet_arm_z),1,
                     FUN = function(x) min(x))
max_dumbbell_accel <- apply(X = cbind(trainData$accel_dumbbell_x,trainData$accel_dumbbell_y,trainData$accel_dumbbell_z),1,
                            FUN = function(x) max(x))
max_dumbbell_mag <- apply(X = cbind(trainData$magnet_dumbbell_x,trainData$magnet_dumbbell_y,trainData$magnet_dumbbell_z),1,
                          FUN = function(x) max(x))
min_dumbbell_mag <- apply(X = cbind(trainData$magnet_dumbbell_x,trainData$magnet_dumbbell_y,trainData$magnet_dumbbell_z),1,
                          FUN = function(x) min(x))
max_glove_gyro <- apply(X = cbind(trainData$gyros_forearm_x,trainData$gyros_forearm_y,trainData$gyros_forearm_z),1,
                        FUN = function(x) max(x))
min_glove_gyro <- apply(X = cbind(trainData$gyros_forearm_x,trainData$gyros_forearm_y,trainData$gyros_forearm_z),1,
                        FUN = function(x) min(x))

trainData.feature <- data.frame(cbind(max_belt_accel,range_belt_accel,var_belt_accel,var_belt_mag,
                           var_arm_accel,max_arm_mag,min_arm_mag,max_dumbbell_accel,max_dumbbell_mag,
                           min_dumbbell_mag,max_glove_gyro,min_glove_gyro))
```

###Cross Validation
Since the original dataset is too large, I'll randomly select 10% of the data based on the proportion of the response variable "classe". Then, I'll use 5-fold cross validation to estimate testing errors for random trees, bagging and boosting, and select the best model to make prediction with the new testing data.


```r
set.seed(1) ## ensure reproducibility
rowind.rdm <- c(sample(1:5580,size = 550), sample(5581:9377,380), sample(9378:12799,343),
            sample(12800:16015,322), sample(16016:19622,360))
trainData <- trainData[rowind.rdm,]
trainData.feature <- trainData.feature[rowind.rdm,]
cv.ind <- sample((rep(1:5, times = length(rowind.rdm)/5)))

library(caret)
rf.err <- vector()
bag.err <- vector()
boost.err <- vector()
for (i in 3:5){
  fit.rf <- train(trainData$classe[cv.ind!=i] ~ ., method = "rf", data = trainData.feature[cv.ind!=i,])
  rf.err[i] = mean(predict(fit.rf,newdata = trainData.feature[cv.ind==i,])!=trainData$classe[cv.ind==i])
  fit.bag <- train(trainData$classe[cv.ind!=i] ~ ., method = "treebag", data = trainData.feature[cv.ind!=i,])
  bag.err[i] = mean(predict(fit.bag,newdata = trainData.feature[cv.ind==i,])!=trainData$classe[cv.ind==i])
  fit.boost <- train(trainData$classe[cv.ind!=i] ~ ., method = "gbm",verbose = FALSE, data = trainData.feature[cv.ind!=i,])
  boost.err[i] = mean(predict(fit.boost.rdm,newdata = trainData.feature[cv.ind==i,])!=trainData$classe[cv.ind==i])
}
mse.rf <- mean(rf.err)
mse.bag <- mean(bag.err)
mse.boost <- mean(boost.err)
```

The MSE_randomforest=0.2072, MSE_bagging=0.2476, MSE_boosting=0.2532. 

###Model Selection
By 5-fold cross validation, I'll choose random forest to predict with the new testing data, because it has the lowest MSE.  

##Prediction with the Testing Data
The expected out of sample error - 0.2 - is estimated by 5-fold cross validation with random forest. Usually, the expected out of sample error will be larger than the in sample error. However, since my training data only contains 10% of the original training data, the results might depend on whether the subset training data is a good representation of the whole dataset.  
Now, apply random forest to the whole training dataset and then apply the model to the [testing data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv), I can get the prediction as follows: A A B A A E D A A A A C E A E E E B B B, with the error rate=0.2, which is quite close to my precious estimation.

