# Practical Machine Learning Course - Week 4 Project
Thao-Truong Nguyen  
January 29, 2017  
## Summary
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Library

```r
library(caret)
library(randomForest)
set.seed(12345)
```

## Get data

```r
# Set data URL
TrainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Download data
if (!file.exists("pml-training.csv")) {
  download.file(url = TrainURL, destfile = "pml-training.csv")
} 
if (!file.exists("pml-testing.csv")) {
  download.file(url = TestURL, destfile = "pml-testing.csv")
} 

# Load data
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Data Cleaning

```r
# Remove variables with Nearly Zero Variance
NZV <- nearZeroVar(training)
training <- training[, -NZV]

# Remove variables contain almost NA
AllNA    <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, AllNA==FALSE]

# Remove first 5 variables, which are identity and time stamp columns
training <- training[, -(1:5)]

# Use 70% of training dataset for train model and 30% for test model
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
```

## Build model
Use Random Forest to build the prediction model:

```r
# build prediction model
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modfitRF <- train(classe ~ ., data=TrainSet, method="rf", trControl=controlRF)
modfitRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.18%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    5 2651    1    1    0 0.0026335591
## C    0    5 2390    1    0 0.0025041736
## D    0    0    5 2247    0 0.0022202487
## E    0    1    0    4 2520 0.0019801980
```

```r
# prediction on Test dataset
predictRF <- predict(modfitRF, TestSet)
confMatRF <- confusionMatrix(predictRF, TestSet$classe)
confMatRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1133    3    0    0
##          C    0    1 1023    6    0
##          D    0    0    0  958    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9968         
##                  95% CI : (0.995, 0.9981)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9959         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9971   0.9938   0.9963
## Specificity            0.9988   0.9994   0.9986   0.9992   1.0000
## Pos Pred Value         0.9970   0.9974   0.9932   0.9958   1.0000
## Neg Pred Value         1.0000   0.9987   0.9994   0.9988   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1925   0.1738   0.1628   0.1832
## Detection Prevalence   0.2853   0.1930   0.1750   0.1635   0.1832
## Balanced Accuracy      0.9994   0.9971   0.9978   0.9965   0.9982
```
Random Forests alogorithm can be used to predict the test set as the accuracy is 99.68% (>99%) and predicted accuracy for the out-of-sample error is 0.28%.

## Predict Test-set
Use Random Forest model to predict the 20 quiz results (testing dataset)

```r
result <- predict(modfitRF,testing)
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Export result data to text file

```r
quiz <- c(1:20)
predictTEST <- data.frame(quiz, result)
write.table(predictTEST, "result.txt", quote = FALSE, row.names = FALSE)
```
