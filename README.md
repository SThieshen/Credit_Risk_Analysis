# Credit Risk Analysis with Supervised Machine Learning: Predicting Credit Risk in Python

## Overview of Project

### Purpose
The purpose of this analysis to utilize different techniques of machine learning to train and evaluate models with unbalanced classes with respect to the issue of credit card risk as generally high-quality loans outnumber risky loans. The six different techniques used are RandomOverSampler, SMOTE Oversampling, ClusterCentroids Undersampling, SMOTEENN Combination, BalancedRandomForestClassifier, and EasyEnsembleClassifier to predict credit risk. The original raw data includes 144 columns and 115,675 observations although variables consisting of only null values will be dropped. The conclusion will include an evaluation of the performance of these models.

## Results

The accuracy score is a tool to measure the performance of a model. It is calculated in the terms of a ratio of the sum of true predictions, either true positives or true negatives, to total predictions. The balanced accuracy score is utilized for imbalanced datasets through taking the average of recall of each class. The precision score, also known as the positive predictive value, is a measure is of the conditional probability of how likely a positive prediction is true given a positive prediction has been made. The recall score, also known as sensitivity, is a measurement of the conditional probability of how likely a positive prediction would be made given that a positive prediction would be true. Below are images of the outputs for each score.

### RandomOverSampler

![RandomOverSampler_Results.png](Resources/RandomOverSampler_Results.png)

* Balanced Accuracy Score is 0.6249984891886339
* Precision Score for High-Risk is 0.01
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.60
* Recall Score for Low-Risk is 0.65
* Recall Score for Total is 0.65

### SMOTE Oversampling

![SMOTE_Oversampling_Results.png](Resources/SMOTE_Oversampling_Results.png)

* Balanced Accuracy Score is 0.6512584051472337
* Precision Score for High-Risk is 0.01
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.64
* Recall Score for Low-Risk is 0.66
* Recall Score for Total is 0.66

### ClusterCentroids Undersampling

![ClusterCentroids_Undersampling_Results.png](Resources/ClusterCentroids_Undersampling_Results.png)

* Balanced Accuracy Score is 0.5103601371413837
* Precision Score for High-Risk is 0.01
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.59
* Recall Score for Low-Risk is 0.43
* Recall Score for Total is 0.44

### SMOTEENN Combination

![SMOTEENN_Combination_Results.png](Resources/SMOTEENN_Combination_Results.png)

* Balanced Accuracy Score is 0.644711676499736
* Precision Score for High-Risk is 0.01
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.72
* Recall Score for Low-Risk is 0.57
* Recall Score for Total is 0.57

### BalancedRandomForestClassifier

![BalancedRandomForestClassifier_Results.png](Resources/BalancedRandomForestClassifier_Results.png)

* Balanced Accuracy Score is 0.6570535621279525
* Precision Score for High-Risk is 0.60
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.32
* Recall Score for Low-Risk is 1.00
* Recall Score for Total is 1.00

### EasyEnsembleClassifier

![EasyEnsembleClassifier_Results.png](Resources/EasyEnsembleClassifier_Results.png)

* Balanced Accuracy Score is 0.9293205140256962
* Precision Score for High-Risk is 0.07
* Precision Score for Low-Risk is 1.00
* Precision Score for Total is 0.99
* Recall Score for High-Risk is 0.92
* Recall Score for Low-Risk is 0.93
* Recall Score for Total is 0.93

## Summary

The models by their balanced accuracy score would have the EasyEnsembleClassifier model as the highest for the measure, signifying that it is the most accurate in producing predictions. After that is the BalancedRandomForestClassifier model, the SMOTE Oversampling model, the SMOTEENN Combination model, RandomOverSampler model, and finally the ClusterCentroids Undersampling model as the least accurate. The precision score for the first four models, RandomOverSampler, SMOTE Oversampling, ClusterCentroids Undersampling, and SMOTEENN Combination, are all the same for each risk category of 0.01 for high-risk, 1.00 for low-risk, and 0.99 for total while the BalancedRandomForestClassifier model, the precision score for high-risk is 0.60, 1.00 for low-risk, and 0.99 for total and the EasyEnsembleClassifier model has a precision score of 0.07 for high-risk, 1.00 for low-risk, and 1.00 for total. What this entails is that the BalancedRandomForestClassifier model has the highest precision score for the high-risk category signifying that it is the model most likely to be correct when it makes a positive prediction while being just as useful for the low-risk category. By recall score, which signifies how likely a positive prediction would be made when it would be true, for the total, the BalancedRandomForestClassifier model has the highest followed by the EasyEnsembleClassifier model, the SMOTE_Oversampling model, the RandomOverSampler model, the SMOTEENN Combination model, and finally the ClusterCentroids Undersampling. I would recommend using the EasyEnsembleClassifier given its performance with highly accurate predictions in general as demonstrated by its balanced accuracy score. Likewise, with the precision score demonstrating its positive predictions would either be just as useful as all others in predicting credit risk for the low-risk category but far more so than all others for high-risk category with the exception of the BalancedRandomForestClassifier model but is more likely to make a positive predicition for the high-risk category, if it would be true to do so, as specified by its recall score for its high-risk category. Although, it also has a slightly lower recall score for the low-risk category, its high recall score for the high-risk category could be useful in investigating this category of borrowers who in general more likely to default in general.
