# CorrYiFeatureSelection

This is an initial description of the current project and its implementation:

The project is about provding a low-complexity algorithm that can be used to select target-influential and nonredundant features from numeric and large-scale datasets.
The algorithm relies on the correlation matrix of features and the target, and is comprised of two main stages that are performed in Cross-Validation:
The first stage involves the removal of features with low correlations with the target based on a threshold. This threshold, referred to as theta_1, is defined as the average of absolute feature-target correlations.
The second stage focuses on eliminating redundant features. Here, redundant features indicate those features that have the same or similar effect or impact on the target variable and should be filtered. 
Thus, the pair-wise correlations of features in the resulted correlation matrix are examined, such that, features with lower correlation with the target are removed from each highly correlated pair.
To identify highly correlated pairs, a second correlation threshold called theta_2 is used, with a default value of 0.68 which can be also tuned as a hyper-parameter. 
After the removal of redundant features in the second stage, the remaining features in each of cross-validation's fold are globally saved.
After cross-validation process is finished, the only common features selected among all folds are returned by the algorithm. 

The algorithm is coded and tested.
The following step before uploading the code is to generate a licence.

The code could be modified by time.

***********************
The authod.
***********************
