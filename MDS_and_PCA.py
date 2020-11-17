import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

### Exercise 1. Performing PCA

## a) Implement PCA
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


## b) Perform PCA on the murder dataset
murder_data = np.loadtxt("murderdata2d.txt")  # load the data
print("performing PCA on the murder data...")

plt.cla()  # remove existing plots to avoid overlap
plt.scatter(murder_data[:, 0], murder_data[:, 1])  # plot the data
plt.xlabel("percent unemployed")
plt.ylabel("murders per annum per 1000000 inhabitants")
mean_x = np.mean(murder_data[:, 0])
mean_y = np.mean(murder_data[:, 1])
plt.scatter(mean_x, mean_y, color='red')  # plot the mean
variances, PCs = pca(murder_data)  # extract the PCs(pointing out of the origin) and variances
PC1 = PCs[:, 0] * np.sqrt(variances[0]) + np.array([mean_x, mean_y])  # scale the PCs by the standard deviations and
PC2 = PCs[:, 1] * np.sqrt(variances[1]) + np.array([mean_x, mean_y])  # move the starting points of the PCs to the mean
plt.plot([mean_x, PC1[0]], [mean_y, PC1[1]], color='green')  # plot PC1
plt.plot([mean_x, PC2[0]], [mean_y, PC2[1]], color='green')  # plot PC2
plt.title("Scatter Plot of Murder Data along with Principal Eigenvectors")
plt.axis("equal")  # make sure the 2 eigenvectors look mutually perpendicular
plt.savefig('Scatter Plot of Murder Data along with Principal Eigenvectors.png', format='png')

print("the murder data PCA finished")

## c) Perform PCA on the pesticide data set
# load the pesticide data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
# split the data into variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, :-1]
YTest = dataTest[:, -1]

# the training set
print("\nperforming PCA on the pesticide data's training set...")
pest_var, pest_PC = pca(XTrain)  # extract the PCs and variances from the training set

# plot the variance versus the PC index
PC_index = list(range(pest_PC.shape[1])) + np.repeat(1, pest_PC.shape[1])
plt.cla()  # remove existing plots to avoid overlap
plt.plot(PC_index, pest_var)
plt.xlabel("PC Index")
plt.axis("auto")  # because it was set to "equal" above
plt.xticks(PC_index)
plt.ylabel("Variance")
plt.subplots_adjust(left=0.18)
plt.title("Variances versus PC Index for the Pesticide Data's Training Set")
plt.savefig("Variances versus PC Index for the Pesticide Data's Training Set.png", format='png')

# plot the cumulative variance versus the PC index
pest_var_norm = pest_var / sum(pest_var)  # normalize the variance along all PCs
pest_var_norm_cumu = []
var_cumu = 0
for var in pest_var_norm:  # create a list of cumulative normalized variances
    var_cumu = var_cumu + var
    pest_var_norm_cumu.append(var_cumu)

plt.cla()  # remove existing plots to avoid overlap
plt.plot(PC_index, pest_var_norm_cumu)
plt.xticks(PC_index)
plt.ylabel("Cumulative Variance")
plt.xlabel("PC Index")
plt.subplots_adjust(left=0.125)
plt.subplots_adjust(right=0.87)
plt.title("Cumulative Variances versus PC Index for the Pesticide Data's Training Set")
plt.savefig("Cumulative Variances versus PC Index for the Pesticide Data's Training Set.png", format='png')

# the test set
print("\nperforming PCA on the pesticide data's test set...")
pest_var_t, pest_PC_t = pca(XTest)  # extract the PCs and variances from the test set

# plot the variance versus the PC index
PC_index_t = list(range(pest_PC_t.shape[1])) + np.repeat(1, pest_PC_t.shape[1])
plt.cla()  # remove existing plots to avoid overlap
plt.plot(PC_index_t, pest_var_t)
plt.xlabel("PC Index")
plt.xticks(PC_index_t)
plt.ylabel("Variance")
plt.subplots_adjust(left=0.18)
plt.title("Variances versus PC Index for the Pesticide Data's Test Set")
plt.savefig("Variances versus PC Index for the Pesticide Data's Test Set.png", format='png')

# plot the cumulative variance versus the PC index
pest_var_norm_t = pest_var_t / sum(pest_var_t)  # normalize the variance along all PCs
pest_var_norm_cumu_t = []
var_cumu_t = 0
for var in pest_var_norm_t:  # create a list of cumulative normalized variances
    var_cumu_t = var_cumu_t + var
    pest_var_norm_cumu_t.append(var_cumu_t)

plt.cla()  # remove existing plots to avoid overlap
plt.plot(PC_index_t, pest_var_norm_cumu_t)
plt.xticks(PC_index_t)
plt.ylabel("Cumulative Variance")
plt.xlabel("PC Index")
plt.subplots_adjust(left=0.125)
plt.subplots_adjust(right=0.87)
plt.title("Cumulative Variances versus PC Index for the Pesticide Data's Test Set")
plt.savefig("Cumulative Variances versus PC Index for the Pesticide Data's Test Set.png", format='png')


### Exercise 2. Visualization in 2D

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


# the training set
pest_projected = mds(XTrain, 2)

print("\nvisualizing the projected training set...")
plt.cla()  # remove existing plots to avoid overlap
plt.scatter(pest_projected[:, 0], pest_projected[:, 1])
plt.title("Pesticide Data's Training Set Projected on to the First 2 PCs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.subplots_adjust(left=0.18)
plt.savefig("Pesticide Data's Training Set Projected on to the First 2 PCs.png", format='png')
print("pesticide data's training set PCA finished")

# the test set
pest_projected_t = mds(XTest, 2)

print("\nvisualizing the projected test set...")
plt.cla()  # remove existing plots to avoid overlap
plt.scatter(pest_projected_t[:, 0], pest_projected_t[:, 1])
plt.title("Pesticide Data's Test Set Projected on to the First 2 PCs")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("Pesticide Data's Test Set Projected on to the First 2 PCs.png", format='png')
print("pesticide data's test set PCA finished")


### Exercise 3. Clustering
print("\nclustering pesticide data's training set...")
startingPoint = np.vstack((XTrain[0, ], XTrain[1, ]))
kmeans = KMeans(n_clusters=2, n_init=1, init=startingPoint, algorithm='full').fit(XTrain)
print("the final cluster centres are:\n", kmeans.cluster_centers_)

