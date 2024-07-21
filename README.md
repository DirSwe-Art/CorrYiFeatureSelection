# CorrYiFeatureSelection

This is an initial description and the implementation of research published at IMSA2024.

Reference paper: 
D. Sweidan (2024). "Correlated, Yet Independent: A Correlation Based Feature Selection Algorithm". 
In proceeding: The IEEE Conference on Intelligent Methods, Systems, and Applications (IMSA2024), July 13-14, 2024, Cairo, Egypt.

The project is about providing a low-complexity algorithm that can be used to select target-influential and nonredundant features from numeric and large-scale datasets.
The algorithm relies on the correlation matrix of features and the target, and is comprised of two main stages that are performed in Cross-Validation:
The first stage involves the removal of features with low correlations with the target based on a threshold. This threshold, referred to as theta_1, is defined as the average of absolute feature-target correlations.
The second stage focuses on eliminating redundant features. Here, redundant features indicate features with the same or similar effect or impact on the target variable and should be filtered. 
Thus, the pair-wise correlations of features in the resulting correlation matrix are examined, such that, features with lower correlation with the target are removed from each highly correlated pair.
To identify highly correlated pairs, a second correlation threshold called theta_2 is used, with a default value of 0.68 which can be also tuned as a hyper-parameter. 
After removing redundant features in the second stage, the remaining features in each cross-validation fold are globally saved.
After the cross-validation process is finished, the algorithm returns the only common features selected among all folds. 

To support the research community, the repository will include detailed documentation and examples to help users understand and apply the algorithm. The algorithm is coded and tested and will be uploaded soon. It could be modified over time, therefore, we encourage others to use and extend our work.

The following step before uploading the code is to generate a license.

***********************
The author.
***********************
