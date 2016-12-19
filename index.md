# Prediction Assignment
_Data Science / Machine Learning / Course Project_   
_Andrey Komrakov_  
_Dec 17 2016_  
_Source files [https://github.com/kolfild26/R_machinelearning_project/tree/gh-pages](https://github.com/kolfild26/R_machinelearning_project/tree/gh-pages)_  

The goal of this report is to build a prediction model, which can be used to estimate a quality of a certain type of physical activity based on a data, taken by accelerometers. Six participants have taken part in the survey. 
More information is available here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

### Setting up and loading packages

The following packages should be installed fisrt:


```r
#install.packages('caret', dependencies=TRUE)
#install.packages('parallel',dependencies=TRUE)
```

Then we could load all required libs.


```r
library(caret)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
```

### Loading a raw data
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv



Download them into R objects:


```r
original_training <- read.csv("pml-training.csv", sep = ",", header = TRUE)
original_testing  <- read.csv("pml-testing.csv" , sep = ",", header = TRUE)
```

### Preparing data for the analysis

First, we skip columns which does not make sense for the analysis:


```r
original_training <- original_training[,-c(1:7)]
original_testing  <- original_testing [,-c(1:7)]
```

Obtained training dataset has 19622 rows and 153 columns:

```r
nrow(original_training);ncol(original_training)
```

```
## [1] 19622
```

```
## [1] 153
```
Obtained testing dataset has 20 rows and 153 columns:

```r
nrow(original_testing);ncol(original_testing)
```

```
## [1] 20
```

```
## [1] 153
```

For the rest of variables we check, whether they have empty, missing or any kind of unsuitable values.
By checking the numbers of NA values, we can notice that if a variable has NA values, a number of such is 19216.


```r
unique(apply(is.na(original_training), 2, sum))[2]
```

```
## [1] 19216
```

```r
unique(apply( (original_training == "" | original_training == "#DIV/0"), 2, sum))[2]
```

```
## [1] 19216
```

It's reasonable to leave them out of a model.


```r
is_not_missing <- apply( !(is.na(original_training)      |
                             original_training == "" |
                             original_training == "#DIV/0"), 2, sum) > 19622 - 19216

original_training <- original_training[, is_not_missing]
original_testing  <- original_testing [, is_not_missing]
```

### Model designing

In order to build our prediction model, we split up the original training data set into two parts, which will be further referred as _**testing**_ and _**training**_. Be sure you first set _**seed()**_ to get the reproducible results.


```r
set.seed(26041986)
```

Using the _**createDataPartition()**_ function from the _**caret**_ package we 


```r
inTrain <- createDataPartition(y= original_training$classe, p = .6)[[1]]

training <- original_training[inTrain, ]
testing  <- original_training[-inTrain,]
```

Let's check, whether some of a potential predictors have very small variance. It could be done through the _**nearZeroVar()**_ function.


```r
length(nearZeroVar(training, allowParallel = TRUE))
```

```
## [1] 0
```

We will use cross validation for each model, so we set up _**trainControl**_ to handle this.


```r
trcontrol <- trainControl(method= "cv", number= 10, allowParallel = TRUE)
```

Next we apply some methods and estimate their quality based on a _**testing**_ data set by comparing to the actual _**classe**_ variable, by the Accuracy characteristic. In most cases "Center" and "scale" preprocessing is used.

To make the algoritms run faster, set up how many cores will be involved in the process.
**Please, be careful with this, as this number depends on your PC. For my machine I allow 4 clusters in parallel. Also, don't forget to roll back, after finishing your analysis by running _stopCluster(cluster)_.**
Refer [for details.](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md#step-4-de-register-parallel-processing-cluster)


```r
cluster  <- makeCluster(4)
registerDoParallel(cluster)
```

#### Rpart Method (Recursive Partitioning and Regression Trees)


```r
modelFitRP <- train(classe ~.,
                  data = training,
                  preProcess= c("center","scale"),
                  method = "rpart")
```


```r
pred_Rpart <- predict(modelFitRP ,testing)
confusionMatrix(testing$classe, pred_Rpart)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.4906959      0.3347382      0.4795757      0.5018230      0.5147846 
## AccuracyPValue  McnemarPValue 
##      0.9999906            NaN
```

#### Random Forest


```r
modelFitRF <- train(classe ~.,
                    data = training,
                    preProcess= c("center","scale"),
                    method = "rf",
                    trainControl = trcontrol
)
```


```r
pred_RF     <- predict(modelFitRF,testing)
confusionMatrix(testing$classe, pred_RF)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9934999      0.9917772      0.9914623      0.9951565      0.2857507 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

#### Bagging (Bootstrap Aggregation)

```r
modelFitBag <- train(classe ~.,
                    data = training,
                    preProcess= c("center","scale"),
                    method = "treebag",
                    trainControl = trcontrol
)
```


```r
pred_Bag <- predict(modelFitBag, testing)
confusionMatrix(testing$classe, pred_Bag)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##    0.983558501    0.979203563    0.980494426    0.986255307    0.284731073 
## AccuracyPValue  McnemarPValue 
##    0.000000000    0.000352851
```

#### GBM (Generalized Boosted Regression Models)


```r
modelFitGBM <- train(classe ~.,
                     data = training,
                     preProcess= c("center","scale"),
                     method = "gbm"
)
```


```r
pred_GBM <- predict(modelFitGBM,testing)
confusionMatrix(testing$classe, pred_GBM)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.644405e-01   9.550039e-01   9.601028e-01   9.684286e-01   2.888096e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   8.536320e-11
```

#### SVM (Support Vector Machine)


```r
modelFitSVM <- train(classe ~.,
                     data = training,
                     preProcess= c("center","scale"),
                     method = "svmRadial",
                     trainControl = trcontrol
)
```


```r
pred_SVM <- predict(modelFitSVM,testing)
confusionMatrix(testing$classe, pred_SVM)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.234005e-01   9.029214e-01   9.172932e-01   9.291909e-01   3.020647e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   1.165688e-58
```

#### Naive Bayes


```r
modelFitNB <- train(classe ~.,
                     data = training,
                     preProcess= c("center","scale"),
                     method = "nb",
                     trainControl = trcontrol
)
```


```r
pred_NB <- predict(modelFitNB,testing)
confusionMatrix(testing$classe, pred_NB)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.379556e-01   6.706436e-01   7.280744e-01   7.476599e-01   2.475147e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00  2.211148e-144
```

Look at the obtained Accuracy characteristics for each model:


```
##          confusionMatrix.testing.classe..pred_Rpart..overall.1.
## Accuracy                                              0.4906959
##          confusionMatrix.testing.classe..pred_RF..overall.1.
## Accuracy                                           0.9934999
##          confusionMatrix.testing.classe..pred_Bag..overall.1.
## Accuracy                                            0.9835585
##          confusionMatrix.testing.classe..pred_GBM..overall.1.
## Accuracy                                            0.9644405
##          confusionMatrix.testing.classe..pred_SVM..overall.1.
## Accuracy                                            0.9234005
##          confusionMatrix.testing.classe..pred_NB..overall.1.
## Accuracy                                           0.7379556
```

It's reasonable to consider _**rpart**_ and _**Naive Bayes**_ models as not competitive, because their accuracies are too low, compared to the others.

#### Combining Predictors
Build a final model, in which we combine 4 successful models Random Forest, Bagging, GBM and SVM.


```r
combDF_training <- data.frame(pred_RF = pred_RF,
                              pred_SVM= pred_SVM,
                              pred_GBM= pred_GBM,
                              pred_Bag= pred_Bag,
                              classe = testing$classe )
                     
combModFit <- train(classe ~.,
                    data = combDF_training,
                    preProcess= c("center","scale"),
                    method = "rf",
                    trainControl = trcontrol
)
```

Check the quality of the result:


```r
combPred_training <- predict(combModFit, combDF_training)
confusionMatrix(combDF_training$classe, combPred_training)$overall[1]
```

```
##  Accuracy 
## 0.9940097
```



As we see, combined model gives the result of 0.994 Accuracy, so it could be applied to a assignment.
First, we need to create a testing dataset.


```r
pred_RF_assignment  <- predict(modelFitGBM, original_testing[,-53])
pred_SVM_assignment <- predict(modelFitGBM, original_testing[,-53])
pred_GBM_assignment <- predict(modelFitGBM, original_testing[,-53])
pred_Bag_assignment <- predict(modelFitGBM, original_testing[,-53])
```


```r
combDF_testing <- data.frame(pred_RF = pred_RF_assignment,
                             pred_SVM= pred_SVM_assignment,
                             pred_GBM= pred_GBM_assignment,
                             pred_Bag= pred_Bag_assignment)
```


```r
pred_Comb_assignment <-  predict(combModFit, combDF_testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

This is a 100% result in the course project prediction quiz.
One can say that combining predictors is a bit far fetched as Random Forest, GBM and Bagging also give 100% result. 
Indeed:

```r
confusionMatrix(pred_RF_assignment,pred_Comb_assignment)$overall[1]
```

```
## Accuracy 
##        1
```

It's not a surprise, as a combined model was just a thousandth part of a percent better on a testing sample. Given only 20 cases in a quiz sample, such small defference gives no benefit. But if we had much more cases, we could win the bid.

