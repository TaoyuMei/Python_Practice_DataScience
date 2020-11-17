import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

# read in the data
dataTrain = np.loadtxt("IDSWeedCropTrain.csv", delimiter=',')
dataTest = np.loadtxt("IDSWeedCropTest.csv", delimiter=',')
# split input variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, :-1]
YTest = dataTest[:, -1]

### Exercise 1. Nearest neighbour classification

def train_and_compute_accuracy(k, XTrain, YTrain, XTest, YTest):
    """input the value of k for k-Nearest Neighbour classifier,
    and the variables and labels of the training set and the test set respectively;
    return the respective accuracy scores of the classifier on the training set and test set"""

    # build and train the classifier
    knn = KNeighborsClassifier(n_neighbors=k)  # the default number of neighbour is 5
    knn.fit(XTrain, YTrain)  # only use the training set to train the model

    # compute the accuracies of the classifier on the training set and the test set respectively
    accTrain = accuracy_score(YTrain, knn.predict(XTrain))
    accTest = accuracy_score(YTest, knn.predict(XTest))
    return accTest, accTrain


accTest, accTrain = train_and_compute_accuracy(k=1, XTrain=XTrain, YTrain=YTrain, XTest=XTest, YTest=YTest)

# print the results
print("when k=1, the accuracy score on the training set is", accTrain)
print("when k=1, the accuracy score on the test set is", accTest)

### Exercise 2. Cross-validation

def CV_for_k(k, XTrain, YTrain):
    """input the value of k for the k-Nearest Neighbour classification
    and the respective variables and labels in the training set;
    perform 5 fold cross-validation and return the average classification error"""
    cv = KFold(n_splits=5)  # create indices for CV
    classification_error = []  # create an empty list to collect the classification errors
    for train, test in cv.split(XTrain):  # loop over CV folds
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]
        accTest, accTrain = train_and_compute_accuracy(k=k, XTrain=XTrainCV, YTrain=YTrainCV,
                                                       XTest=XTestCV, YTest=YTestCV)
        classification_error.append((1 - accTest) * len(YTestCV))  # compute and collect the classification error
    avg_class_err = np.mean(classification_error)
    return avg_class_err


# find a good value for k from {1, 3, 5, 7, 9, 11}
k_list = [1, 3, 5, 7, 9, 11]
for k in k_list:
    avg_class_err = CV_for_k(k=k, XTrain=XTrain, YTrain=YTrain)
    print("the average classification error for k=", k, "is", avg_class_err)

### Exercise 3. Evaluation of classification performance

# set the value of k as the k-best, which is 3
accTest, accTrain = train_and_compute_accuracy(k=3, XTrain=XTrain, YTrain=YTrain, XTest=XTest, YTest=YTest)

# print the results
print("when k=3, the accuracy score on the training set is", accTrain)
print("when k=3, the accuracy score on the test set is", accTest)

### Exercise 4. Data normalization

# center and normalize the training data and the test data
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
scaler = preprocessing.StandardScaler().fit(XTest)
XTestN = scaler.transform(XTest)

# repeat the model selection in exercise 2
k_list = [1, 3, 5, 7, 9, 11]
for k in k_list:
    avg_class_err = CV_for_k(k=k, XTrain=XTrainN, YTrain=YTrain)  # use the normalized training data variables XTrainN
    print("after data normalization, the average classification error for k=", k, "is", avg_class_err)

# repeat the exercise 3 using normalized data
# set the value of k as the k-best, which is still 3; use transformed variables XTrainN and XTestN
accTest, accTrain = train_and_compute_accuracy(k=3, XTrain=XTrainN, YTrain=YTrain, XTest=XTestN, YTest=YTest)
# print the results
print("after data normalization, when k=3, the accuracy score on the training set is", accTrain)
print("after data normalization, when k=3, the accuracy score on the test set is", accTest)
