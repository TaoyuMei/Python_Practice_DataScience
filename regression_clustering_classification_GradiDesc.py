import numpy as np
import numpy.matlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sympy as sp

### Exercise 1. Bayesian Statistics

# no code needed, see the report

### Exercise 2. Using Linear Regression

# load and split the red wine data training set
redwine_train = np.loadtxt("redwine_training.txt")
physiochemical = redwine_train[:, 0:11]
quality = redwine_train[:, 11]

# define the multivariate linear regression function
def multivarlinreg(X, y):
    """
    input the N*d independent variable matrix and the N dimensional dependent variable vector;
    output an D+1 dimensional vector of regression coefficients
    """

    X = np.matrix(X)
    ones_column = numpy.matlib.repmat(1, X.shape[0], 1)
    X = np.column_stack((ones_column, X))  # add a column filled with 1 to the beginning of the data matrix
    y = np.matrix(y).T  # transfer y to a column vector (single-vector matrix)
    one = np.dot(X.T, X)
    two = np.dot(np.linalg.pinv(one), X.T)  # np.dot(X.T, X) produces a singular matrix, cannot use inv
    w = np.dot(two, y)
    w = np.array(w)
    return w


# call the multivarlinreg function on the first feature and the wine quality
# the function can accept both np.matrix and mp.array as input
# but when the X only has one column, using matrix make it easier to keep it as a column rather than a row
w_first = multivarlinreg(np.matrix(physiochemical)[:, 0], quality)
print("when the input data only contain the first feature, the weights are:\n", w_first)

# run the regression function on all features
w_all = multivarlinreg(physiochemical, quality)
print("\nfor all features, the weights are:\n", w_all)

### Exercise 3. Evaluating Linear Regression

# define the rmse function
def rmse(f, t):
    """
    param f: the predicted values of the dependent output variable, an N-dimensional vector, as a numpy array
    param t: the ground truth values of dependent output variable, an N-dimensional vector, as a numpy array
    return r: the root mean square error (rmse) as a 1 x 1 numpy array
    """
    r = np.sqrt(np.mean((t - f) ** 2))
    return r


# load and split the red wine data test set
redwine_test = np.loadtxt("redwine_testing.txt")
physiochemical_test = redwine_test[:, 0:11]
quality_test = redwine_test[:, 11]

# add a column filled with 1 to the beginning of the data matrix
ones_column = numpy.matlib.repmat(1, physiochemical_test.shape[0], 1)
physiochemical_test = np.column_stack((ones_column, physiochemical_test))

# build a model with the first feature and compute the RMSE for the model
w_first = multivarlinreg(np.matrix(physiochemical)[:, 0], quality)
f_first = np.array(np.dot(np.matrix(physiochemical_test)[:, 0:2], w_first).T)[0]  # predict
rmse_first = rmse(f_first, quality_test)
print("\nwhen the input data only contain the first feature, the RMSE is", rmse_first)

# build a model with all features and compute the RMSE for the model
w_all = multivarlinreg(physiochemical, quality)
f_all = np.dot(physiochemical_test, w_all).T[0]  # predict
rmse_all = rmse(f_all, quality_test)
print("\nwhen using all features, the RMSE is", rmse_all)

### Exercise 4. Random forest & normalization

# no code needed, see the report

### Exercise 5. Applying random forest

# load and split the pesticide data
pesticide_train = np.loadtxt("IDSWeedCropTrain.csv", delimiter=',')
pesticide_train_X = pesticide_train[:, :-1]
pesticide_train_Y = pesticide_train[:, -1]

pesticide_test = np.loadtxt("IDSWeedCropTest.csv", delimiter=',')
pesticide_test_X = pesticide_test[:, :-1]
pesticide_test_Y = pesticide_test[:, -1]

# train the random forest classifier with 50 trees in the forest using the training set
rfc = RandomForestClassifier(n_estimators=50).fit(pesticide_train_X, pesticide_train_Y)

# test the classifier using the test set
rfc_accuracy = accuracy_score(pesticide_test_Y, rfc.predict(pesticide_test_X))
print("\nthe prediction accuracy for the random forest classifier is", rfc_accuracy)

### Exercise 6. Gradient descent & learning rates

x = sp.Symbol('x')
y = sp.exp(-x/2) + 10 * x ** 2  # write the original function as symbols
y_prime = y.diff(x)  # compute the derivative function
f_prime = sp.lambdify(x, y_prime, 'numpy')  # convert the symbolic derivative function to a callable function
f = sp.lambdify(x, y, 'numpy')  # convert the symbolic original function to a callable function

def grad_desc(learning_rate, starting_point, f, f_prime):
    """
    input: the learning rate, the starting point, the callable original function, and the callable derivative function,
    produce plots showing:
    1) tangent lines and gradient descent steps for the first 3 iterations;
    2) gradient descent steps (without tangent lines) for the first 10 iterations;
    return:
    1) the final number of iteration;
    2) the value of the original function at the final iteration (expected to be the minimum);
    """

    # initialize and set parameters for convergence check
    point = starting_point
    num_iter = 1  # number of iteration
    min_mag_grad = 10 ** (-10)  # the minimum of the magnitude of the gradient
    max_iter = 10000  # the maximum number of iteration
    converge = False
    x_10 = []  # record the range of the x axis after 10 iterations to facilitate plotting the curve of the function

    # iterate
    plt.cla()
    while converge == False:
        grad = np.nan_to_num(f_prime(point))  # use np.nan_to_num() to prevent the overflow problem from producing nan

        if num_iter <= 11:
            x_10.append(point)

        # visualize the tangent lines and gradient descent steps for the first 3 iterations
        if num_iter <= 3:
            plt.figure(1)  # figure 1
            t2 = [point, f(point)]  # 3 points for drawing the tangent line
            t1 = [point - learning_rate * -grad / 4, f(point) + learning_rate * grad * grad / 4]
            t3 = [point + learning_rate * -grad / 4, f(point) - learning_rate * grad * grad / 4]
            plt.plot([t1[0], t2[0], t3[0]], [t1[1], t2[1], t3[1]], color='black')

            if grad * f_prime(point + learning_rate * -grad) >= 0:
                p1 = [point, f(point)]  # 3 points for drawing the step
                p2 = [point + learning_rate * -grad, f(point)]
                p3 = [point + learning_rate * -grad, f(point + learning_rate * -grad)]
                plt.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='blue')
            else:  # if the gradient change its sign after this step
                p1 = [point, f(point)]
                p2 = [point + learning_rate * -grad, f(point + learning_rate * -grad)]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue')

        # visualize the gradient descent steps (without tangent lines) for the first 10 iterations
        if num_iter <= 10:
            plt.figure(2)  # figure 2
            if grad * f_prime(point + learning_rate * -grad) >= 0:
                p1 = [point, f(point)]
                p2 = [point + learning_rate * -grad, f(point)]
                p3 = [point + learning_rate * -grad, f(point + learning_rate * -grad)]
                plt.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color='blue')
            else:  # if the gradient change its sign after this step
                p1 = [point, f(point)]
                p2 = [point + learning_rate * -grad, f(point + learning_rate * -grad)]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue')

        point = point + learning_rate * -grad
        num_iter += 1
        if (num_iter > max_iter or abs(grad) < min_mag_grad):
            converge = True

    # add the curve of the original function to the 2 figures and save the 2 respectively
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    f1_name = "the tangent lines and gradient descent steps for the first 3 iterations"
    f1_name1 = f1_name + "\nwith a learning rate " + str(learning_rate)
    plt.title(f1_name1)
    f1_name2 = f1_name + " with a learning rate " + str(learning_rate)
    plt.savefig(f1_name2 + ".png", format='png')
    plt.cla()

    plt.figure(2)
    xx = np.linspace(min(x_10), max(x_10), 50)
    yy = f(xx)
    plt.plot(xx, yy, color='green')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    f2_name = "the gradient descent steps for the first 10 iterations"
    f2_name1 = f2_name + "\nwith a learning rate " + str(learning_rate)
    plt.title(f2_name1)
    f2_name2 = f2_name + " with a learning rate " + str(learning_rate)
    plt.savefig(f2_name2 + ".png", format='png')
    plt.cla()

    value = f(point)
    num_iter -= 1

    return num_iter, value


print("\n")
for lr in [0.1, 0.01, 0.001, 0.0001]:
    iteration_number, function_value = grad_desc(learning_rate=lr, starting_point=1, f=f, f_prime=f_prime)
    st1 = "when the learning rate=" + str(lr)
    st2 = ", the algorithm converges after " + str(iteration_number)
    st3 = " iterations when the value of the function is " + str(function_value)
    print(st1 + st2 + st3)


### Exercise 7. Logistic regression implementation

# load and split the 2 Iris data sets
iris1_train = np.loadtxt("Iris2D1_train.txt")
iris1_train_X = iris1_train[:, :-1]
iris1_train_Y = iris1_train[:, -1]

iris1_test = np.loadtxt("Iris2D1_test.txt")
iris1_test_X = iris1_test[:, :-1]
iris1_test_Y = iris1_test[:, -1]

iris2_train = np.loadtxt("Iris2D2_train.txt")
iris2_train_X = iris2_train[:, :-1]
iris2_train_Y = iris2_train[:, -1]

iris2_test = np.loadtxt("Iris2D2_test.txt")
iris2_test_X = iris2_test[:, :-1]
iris2_test_Y = iris2_test[:, -1]

# make a scatter plot for each of the 4 data sets
def scatter_plot(name, X, Y):
    """input the name of the data set, the data matrix X containing independent variables, and the vector Y containing labels;
    make a scatter plot in which the colour of the points corresponding its label"""

    # collect data points with label 0 or 1 separately
    points_0 = []
    points_1 = []
    for i in range(len(Y)):
        if Y[i] == 0:
            points_0.append(X[i, :])
        elif Y[i] == 1:
            points_1.append(X[i, :])
    points_0 = np.array(points_0)
    points_1 = np.array(points_1)

    # plot points in 2 classes
    plt.cla()
    plt.scatter(points_0[:, 0], points_0[:, 1], label='class 0', color='yellow')
    plt.scatter(points_1[:, 0], points_1[:, 1], label='class 1', color='green')
    plt.legend()
    plt.title("scatter plot of the data set " + name)
    plt.savefig("scatter plot of the data set " + name + ".png", format='png')
    return 0


scatter_plot("Iris2D1_train", iris1_train_X, iris1_train_Y)
scatter_plot("Iris2D1_test", iris1_test_X, iris1_test_Y)
scatter_plot("Iris2D2_train", iris2_train_X, iris2_train_Y)
scatter_plot("Iris2D2_test", iris2_test_X, iris2_test_Y)

# define the gradient function
def gradient(X, Y, w):
    """inout the data matrix, label vector, and parameter vector, return the gradient"""

    every_line = []
    for n in range(len(Y)):
        # split the long expression
        every_line.append(Y[n] * X[n, :] / (1 + np.exp(Y[n] * w.T @ X[n, :])))  # @ means dot product
    every_line = np.array(every_line)
    gradi = -np.mean(every_line, 0)
    return gradi


# define the logistic regression function
def logi_regr(train_X, train_Y, test_X):
    """input the training set data matrix, the training set label vector, and the test set data matrix;
    return:
    1) a vector w of the 3 parameters of the affine linear model;
    2) a vector of predicted labels for the training set
    3) a vector of predicted labels for the test set"""

    # initialize
    eta = 0.1  # the learning rate / step size
    np.random.seed(0)
    w = -0.01 * np.random.randn(3)
    iter_num = 1
    min_mag_grad = 10 ** (-10)  # the minimum of the magnitude of the gradient
    max_iter = 10000  # the maximum number of iteration
    converge = False

    # convert the label from 0/1 to -1/+1
    train_Y = (train_Y - 0.5) * 2

    # add a column of ones at the beginning of the 2 data matrices
    ones_column_train = numpy.matlib.repmat(1, train_X.shape[0], 1)
    train_X = np.column_stack((ones_column_train, train_X))
    ones_column_test = numpy.matlib.repmat(1, test_X.shape[0], 1)
    test_X = np.column_stack((ones_column_test, test_X))
    print("\nthe initial gradient is", gradient(train_X, train_Y, w))

    # iterate
    while converge == False:
        gradie = gradient(train_X, train_Y, w)  # compute the gradient using the function defined above
        w = w - eta * gradie
        iter_num += 1
        if(iter_num >= max_iter or np.linalg.norm(gradie) < min_mag_grad):
            converge = True

    print("the final gradient is", gradie)
    print("\nthe iteration number for the logistic regression is", iter_num)

    # predict the labels
    s_train = train_X @ w.T
    theta_train = 1 / (1 + np.exp(-s_train))  # theta is the probability that y = 1, which is the same before conversion
    theta_train = np.rint(theta_train)  # if theta <= 0.5, it is assigned the value 0, the predicted label is also 0

    s_test = test_X @ w.T
    theta_test = 1 / (1 + np.exp(-s_test))
    theta_test = np.rint(theta_test)

    return theta_train, theta_test, w

# for iris 1
iris1_train_theta, iris1_test_theta, iris1_w = logi_regr(iris1_train_X, iris1_train_Y, iris1_test_X)
iris1_train_accu = accuracy_score(iris1_train_Y, iris1_train_theta)
iris1_test_accu = accuracy_score(iris1_test_Y, iris1_test_theta)
print("\nthe train and test accuracy of Iris2D1 data set are:\n", iris1_train_accu, "\n", iris1_test_accu)
print("the 3 parameters of the affine linear model are", iris1_w, "\n")

# for iris 2
iris2_train_theta, iris2_test_theta, iris2_w = logi_regr(iris2_train_X, iris2_train_Y, iris2_test_X)
iris2_train_accu = accuracy_score(iris2_train_Y, iris2_train_theta)
iris2_test_accu = accuracy_score(iris2_test_Y, iris2_test_theta)
print("\nthe train and test accuracy of Iris2D2 data set are:\n", iris2_train_accu, "\n", iris2_test_accu)
print("the 3 parameters of the affine linear model are", iris2_w, "\n")

### Exercise 8. Logistic regression loss-gradient

# no code needed, see the report

### Exercise 9. Clustering and classification I

# load the MNIST data set
mnist_digit = np.loadtxt("MNIST_179_digits.txt")
mnist_label = np.loadtxt("MNIST_179_labels.txt")

# a) k-means clustering
def KMeans_and_CalPro(data, label):
    """input the data matrix and label vector, perform k-means clustering,
    and calculate the proportion of 1s, 7s and 9s in each cluster
    return the coordinates of the 3  cluster centres"""

    # prepend the label to the data as the first column, which is meaningless
    data = np.column_stack((label.T, data))

    # specifying starting point seems to worsen the result
    kmeans = KMeans(n_clusters=3, n_init=10, algorithm='full').fit(data[:, 1:])  # train the clusterer without the first column
    label_predicted = kmeans.predict(data[:, 1:])

    # calculate the proportion of 1s, 7s, and 9s in cluster 0, 1, and 2
    c_0 = []
    c_1 = []
    c_2 = []

    for j in range(len(label_predicted)):
        if label_predicted[j] == 0:
            c_0.append(label[j])
        elif label_predicted[j] == 1:
            c_1.append(label[j])
        elif label_predicted[j] == 2:
            c_2.append(label[j])

    c_012 = [c_0, c_1, c_2]
    for k in range(len(c_012)):
        l_1 = 0
        l_7 = 0
        l_9 = 0
        for label in c_012[k]:
            if label == 1:
                l_1 += 1
            elif label == 7:
                l_7 += 1
            elif label == 9:
                l_9 += 1
        p_1 = l_1 / len(c_012[k])
        p_7 = l_7 / len(c_012[k])
        p_9 = l_9 / len(c_012[k])
        print("in the cluster", k, "the proportion of 1s, 7s, and 9s are", p_1, p_7, "and", p_9, "respectively")

    return kmeans.cluster_centers_


mnist_cluster_centres = KMeans_and_CalPro(mnist_digit, mnist_label)

def plot_a_centre(centre):
    """input a numpy array of pixels.
    28 * 28 = 784 pixels on total, organized row-wise, beginning from the top-left of the original image.
    generate a 2D plot with each pixel in its original position.
    the gray level of each point is proportional to its pixel value
    """

    ide = 0
    for pixel in centre:
        pixel_x = [ide % 28]
        pixel_y = [27 - (ide // 28)]
        colours = plt.get_cmap('Blues')
        plt.scatter(pixel_x, pixel_y, c=colours(pixel / 255))
        plt.axis('equal')
        ide += 1
    return 0


# plot the 3 cluster centres
i = 0
for ct in mnist_cluster_centres:
    plt.cla()
    plot_a_centre(ct)
    plt.title("cluster centre " + str(i))
    plt.savefig("cluster centre " + str(i) + ".png", format='png')
    i += 1

# b) k-NN classification
# collect and modify useful functions defined in assignment 2
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
        classification_error.append(1 - accTest)  # compute and collect the classification error
    avg_class_err = np.mean(classification_error)
    return avg_class_err


# find the k-best using cross-validation
k_list = [1, 3, 5, 7, 9, 11]
print("\n")
for k in k_list:
    avg_class_err = CV_for_k(k=k, XTrain=mnist_digit, YTrain=mnist_label)
    print("the average classification error for k=" + str(k), "is", avg_class_err)

# calculate the test accuracy for k-best=1 on the whole data set
acc_1, acc_2 = train_and_compute_accuracy(1, mnist_digit, mnist_label, mnist_digit, mnist_label)
# use the whole data set as a 'pretended' test set, as there is only one data set
print("\nthe test accuracy for k-best is", acc_1)

### Exercise 10. Clustering and classification after dimensionality reduction

# collect useful functions for PCA from previous assignments
def pca(data):
    """input a data matrix in which each column corresponds to a coordinate and each row corresponds to a data point;
    output the eigenvalues in a vector (numpy array) in descending order
    and a matrix where each column is an eigenvector of the corresponding eigenvalue"""

    # center the data
    data_mean = np.mean(data, 0)  # 0 means calculate the means of each column
    data_mean_matrix = numpy.matlib.repmat(data_mean, data.shape[0], 1)
    data_centered = data - data_mean_matrix

    # construct the input matrix for svd
    data_centered = data_centered / np.sqrt(data.shape[0] - 1)

    # use singular value decomposition
    (u, S, PC) = np.linalg.svd(data_centered)  # S is a vector of standard deviation rather than a diagonal matrix
    eigenvalues = S * S  # compute the variances which are already in descending order
    PC = PC.T  # make sure each column in the matrix is an eigenvector (the length ia already 1)
    return eigenvalues, PC


def mds(data, d):
    """input a data matrix with each column corresponding to a coordinate and each row corresponding to a data point,
    and the number of PCs to be selected;
    output an N * d data matrix containing the d coordinates of N data points projected on to the top d PCs
    """
    if d > data.shape[1]:
        print("you specify too much PCs, the data have only", data.shape[1], "dimensions.")

    _, PC = pca(data)  # extract the PCs
    PC_selected = PC[:, 0:d]  # select first d PCs

    # center the data
    data_mean = np.mean(data, 0)  # 0 means calculate the means of each column
    data_mean_matrix = numpy.matlib.repmat(data_mean, data.shape[0], 1)
    data_centered = data - data_mean_matrix

    data_projected = np.dot(PC_selected.T, data_centered.T).T  # project the data on to the selected PCs

    return data_projected


# a) perform PCA and plot the cumulative variance (in %) w.r.t. the principal components
mnist_var, mnist_PC = pca(mnist_digit)

PC_index = list(range(mnist_PC.shape[1])) + np.repeat(1, mnist_PC.shape[1])  # create PC indices beginning from 1

mnist_var_norm = mnist_var / sum(mnist_var)  # normalize the variance along all PCs
mnist_var_norm_cumu = []
var_cumu = 0
for var in mnist_var_norm:  # create a list of cumulative normalized variances
    var_cumu = var_cumu + var
    mnist_var_norm_cumu.append(var_cumu)

plt.cla()  # remove existing plots to avoid overlap
plt.plot(PC_index, mnist_var_norm_cumu)
plt.axis('auto')
plt.ylabel("Cumulative Variance")
plt.xlabel("PC Index")
plt.subplots_adjust(left=0.125)
plt.subplots_adjust(right=0.87)
plt.title("Cumulative Variances versus PC Index for the MNIST Data Set")
plt.savefig("Cumulative Variances versus PC Index for the MNIST Data Set.png", format='png')

# b)
# project the MNIST data on the first 20 and 200 PCs and run k-means clustering
mnist_digit_projected_20 = mds(mnist_digit, 20)
mnist_digit_projected_200 = mds(mnist_digit, 200)

print("\nfor the first 20 PCs:")
mnist_cluster_centres_20 = KMeans_and_CalPro(mnist_digit_projected_20, mnist_label)
print("\nfor the first 200 PCs:")
mnist_cluster_centres_200 = KMeans_and_CalPro(mnist_digit_projected_200, mnist_label)

# visualize the 3 cluster centres for first 20 and 200 PCs respectively

# recover the digit images in 28 * 28 pixels
mnist_cluster_centres_20 = mnist_cluster_centres_20 @ mnist_PC[:, 0:20].T  # reverse transform
mnist_cluster_centres_200 = mnist_cluster_centres_200 @ mnist_PC[:, 0:200].T  # reverse transform

for i in range(mnist_cluster_centres_20.shape[0]):
    plt.cla()
    plot_a_centre(mnist_cluster_centres_20[i])
    plt.title("the Cluster Centre " + str(i) + " for the First 20 PCs")
    plt.savefig("the Cluster Centre " + str(i) + " for the First 20 PCs.png", format='png')

for i in range(mnist_cluster_centres_200.shape[0]):
    plt.cla()
    plot_a_centre(mnist_cluster_centres_200[i])
    plt.title("the Cluster Centre " + str(i) + " for the First 200 PCs")
    plt.savefig("the Cluster Centre " + str(i) + " for the First 200 PCs.png", format='png')

# c) train k-NN classifier and perform n-fold validation on the MNIST data projected on the first 20 and 200 PCs

# find the k-best using cross-validation
k_list = [1, 3, 5, 7, 9, 11]
print("\nfor the first 20 PCs:")
for k in k_list:
    avg_class_err = CV_for_k(k=k, XTrain=mnist_digit_projected_20, YTrain=mnist_label)
    print("the average classification error for k=" + str(k), "is", avg_class_err)
# calculate the test accuracy for k-best=1 on the whole data set
acc_1, _ = train_and_compute_accuracy(1, mnist_digit_projected_20, mnist_label, mnist_digit_projected_20, mnist_label)
# use the whole data set as a 'pretended' test set
print("the test accuracy for k-best is", acc_1)

# find the k-best using cross-validation
k_list = [1, 3, 5, 7, 9, 11]
print("\nfor the first 200 PCs:")
for k in k_list:
    avg_class_err = CV_for_k(k=k, XTrain=mnist_digit_projected_200, YTrain=mnist_label)
    print("the average classification error for k=" + str(k), "is", avg_class_err)
# calculate the test accuracy for k-best=1 on the whole data set
acc_1, _ = train_and_compute_accuracy(1, mnist_digit_projected_200, mnist_label, mnist_digit_projected_200, mnist_label)
# use the whole data set as a 'pretended' test set
print("the test accuracy for k-best is", acc_1)
