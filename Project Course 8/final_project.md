---
title: "Final Project: Buil a Prediction Model"
author: "Author: Hiba"
date: "11/04/2020"
output:
  html_document: 
    keep_md: yes
---

## Overview
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
A prediction model is built to predict which of these 5 ways is the exercice being performed. The final model use a Random Forest method with 99.03% accuracy on the validation set.


## Data Processing  
  Let's first read the raw files and useful packages:  


```r
library(caret)
pml_training <- read.csv('pml-training.csv')
pml_testing <- read.csv('pml-testing.csv')
```
 **Features Selection:** Since the prediction model will be used to predict 20 different test cases from the `pml_testing` data, we need to take into account what kind of features it has, and to do so, we will select features which matches the columns containing no NA values in the `pml_testing`data:




```r
features <- colnames(pml_testing)[colSums(is.na(pml_testing)) == 0]
features <- features[c(-(1:10),-60)]
sub_pml_training <- pml_training[,c(features,"classe")]
```

**Cross Validation:** Before we build our model and directly apply it to the testing data, we will split the training data into *sub_train* (80%) and *sub_test* (20%):


```r
set.seed(23432)
inTrain = createDataPartition(sub_pml_training$classe, p = 0.8)[[1]]
sub_train = sub_pml_training[ inTrain,]  # 15699 obs. of  50 variables
sub_test = sub_pml_training[ -inTrain,]  #3923 obs. of  50 variables
```

### Prediction with Trees
We will first try this model, which iteratively splits variables into groups (effectively constructing decision trees) to produces a nonlinear model and classify our observations into a specific manner of doing the exercise:

```r
treemod <- train(classe ~ ., data=sub_train, method="rpart")
library(rattle )
fancyRpartPlot(treemod$finalModel, main="Rattle plot of the decision tree classification model",caption="")
```

![](final_project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->


```r
confusionMatrix(predict(treemod, sub_test), sub_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1016  331  316  263  171
##          B   17  253   17  135  135
##          C   64   83  296   97  102
##          D   15   92   55  148   99
##          E    4    0    0    0  214
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4912         
##                  95% CI : (0.4754, 0.507)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3346         
##                                          
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33333  0.43275  0.23017  0.29681
## Specificity            0.6149  0.90392  0.89318  0.92043  0.99875
## Pos Pred Value         0.4845  0.45422  0.46106  0.36186  0.98165
## Neg Pred Value         0.9452  0.84967  0.88174  0.85913  0.86316
## Prevalence             0.2845  0.19347  0.17436  0.16391  0.18379
## Detection Rate         0.2590  0.06449  0.07545  0.03773  0.05455
## Detection Prevalence   0.5345  0.14198  0.16365  0.10426  0.05557
## Balanced Accuracy      0.7626  0.61863  0.66296  0.57530  0.64778
```
As the accuracy of the decicison tree model is small (below 50%). We will try another model:

### Random Forest
As an extension of bagging on classification trees, we will try a Random Forest model, with 100 trees. Instead of the `train` function from caret package, we use the `randomForest` package since it's faster:

```r
library(randomForest)
modrf <- randomForest(classe~., data=sub_train, ntree=100, do.trace=F)
confusionMatrix(predict(modrf, sub_test), sub_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    7    0    0    0
##          B    2  745    8    0    0
##          C    0    5  675   13    1
##          D    0    1    1  630    1
##          E    0    1    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9898          
##                  95% CI : (0.9861, 0.9927)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9871          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9816   0.9868   0.9798   0.9972
## Specificity            0.9975   0.9968   0.9941   0.9991   0.9997
## Pos Pred Value         0.9938   0.9868   0.9726   0.9953   0.9986
## Neg Pred Value         0.9993   0.9956   0.9972   0.9960   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1899   0.1721   0.1606   0.1833
## Detection Prevalence   0.2858   0.1925   0.1769   0.1614   0.1835
## Balanced Accuracy      0.9979   0.9892   0.9905   0.9894   0.9985
```
The accuracy of this model is very high (99.03%).
We will keep this model.  
  
    
    

### Prediction on the 20 observations

We can now confidently use the Random Forest model on the `pml_testing` dataset to predict the manner in which the exercise was done for the 20 observations:




```r
predict(modrf, pml_testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



