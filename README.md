# SVM_SimulatedAnnealing
Parameter Optimization for Support Vector Machine  Based on Simulated Annealing.

-OBJECTIVES

Using the simulated annealing method, our goal is to maximise the ideal parameter values and feature subset at the same time without sacrificing the accuracy of the SVM classification.
In addition, the project intends to cover objectives include: 
1) to develop parameter optimization algorithm based on SA for classification problem using SVM
2) to evaluate the performance of Simulated Annealing algorithm in looking for the best parameter 

-SCOPES

SVMs were initially designed to handle classification issues, but they may also be modified to address nonlinear regression issues. In order to make optimal use of SVM's performance, it is necessary to choose the appropriate parameters for the kernel function. Studies have shown that the penalty parameter C and the kernel function parameter γ, gamma are two factors that significantly impact the performance of an SVM.
However, their generalisation performance is occasionally much below the desired level due to several constraints in actual applications. Therefore, SA-SVM was proposed based on the parameter optimization of SVM by simulated annealing and only the standard implementation of SA have been used in parameter searching for this project.
In our experiments, to validate the proposed SA-SVM algorithm's performance in classification, researcher conducted the tests on five data sets that were taken from the UCI Machine Learning Repository. The selections of the dataset are only limited to those in UCI Machine Learning Repository. 
To determine whether or not the technique that researcher suggested would be effective, researcher implemented the methodology using Python 3 and PyCharm IDE as the software platform. This allowed us to evaluate the effectiveness of our recommended procedure.
For evaluating the effectiveness of the suggested performances, the performance measurements were calculated based on accuracy of the classification during testing, by utilising methods like recall, precision, F-measure, and overall accuracy. Furthermore, the progress of the search will be reviewed as a line plot that displays the change in the evaluation of the best solution each time there is an improvement.

-DATASETS

<img width="594" alt="image" src="https://user-images.githubusercontent.com/45988034/219648439-7a9bc80b-f971-41c6-975a-ffa069a3b5af.png">

-RESULTS

| SVM WITH DEFAULT PARAMETER |

<img width="620" alt="image" src="https://user-images.githubusercontent.com/45988034/219649389-b3722976-7f70-441c-85d1-cdc5503929e7.png">

| SVM WITH GRID SEARCH |

<img width="645" alt="image" src="https://user-images.githubusercontent.com/45988034/219648673-4bd8244d-fe71-401e-bd57-343c43f8d9fd.png">

| SVM WITH SIMULATED ANNEALING |

<img width="606" alt="image" src="https://user-images.githubusercontent.com/45988034/219648860-44eb4343-e41f-406f-842f-77118d2b9d89.png">

| SVM PARAMETERS OPTIMIZATION USING GRID SEARCH |

<img width="612" alt="image" src="https://user-images.githubusercontent.com/45988034/219649061-552ee9dc-30d7-4f6b-bcac-a3974a9f4f30.png">

| SVM PARAMETERS OPTIMIZATION USING SIMULATED ANNEALING |

<img width="602" alt="image" src="https://user-images.githubusercontent.com/45988034/219649190-a19c586b-bbf4-4f9c-99f6-b254ac4676b6.png">

| GRID SEARCH AND SA COMPARISON |

<img width="599" alt="image" src="https://user-images.githubusercontent.com/45988034/219649524-6e2ccf44-8bb4-437a-a579-3de6c4639a85.png">

-DISCUSSION

Based on the results provided, it appears that the simulated annealing algorithm was able to find better parameters for the SVM classifier than the conventional grid search method. This is evident by the higher precision, recall, f-score, and accuracy values for most of the datasets when using the simulated annealing algorithm. This suggests that the simulated annealing algorithm is able to avoid being trapped in local minima and produce better performance compared to the normal SVM method.

In terms of this research objective of developing a parameter optimization algorithm based on SA for classification problems using SVM, it seems that the Simulated Annealing algorithm is able to find good parameters and optimal parameters for the classifier.

In terms of evaluating the performance of the Simulated Annealing algorithm in finding the best parameters, the results suggest that the Simulated Annealing algorithm is able to find optimal parameters for the classifiers. The high accuracy rates achieved for all the datasets indicate that the model is performing well on these datasets.






