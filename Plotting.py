import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.cluster import KMeans

### Exercise 1. Plotting cell shapes

diatoms_data = np.loadtxt("diatoms.txt")
def plot_a_cell(cell, colour='b'):
    """input a numpy array of the 180 coordinates of the 90 landmark points in 'x1, y1, x2, y2...x90, y90' manner;
    generate a plot of landmark points and interpolating between subsequent landmark points;
    colour can also be specified, the default is blue;
    """

    cell_x = []
    cell_y = []
    i = 0
    for coor in cell:
        if i % 2 == 0:
            cell_x.append(coor)
        else:
            cell_y.append(coor)
        i += 1
    plt.plot(cell_x, cell_y, c=colour)
    plt.axis('equal')


print("begin to plot cell(s)")
# plot a cell
plt.title("the Plot of a Cell")
plot_a_cell(diatoms_data[0, :], colour='b')
plt.savefig("the Plot of a Cell.png", format="png")

# plot all cells
plt.cla()  # remove existing plots to avoid overlap
plt.title("the Plot of All Cells")
for i in range(diatoms_data.shape[0]):
    co = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][np.random.randint(8)]  # randomly choose a colour
    plot_a_cell(diatoms_data[i, :], colour=co)
plt.savefig("the Plot of All Cells.png", format="png")
print("finish plotting cell(s)")

### Exercise 2. Visualizing variance in visual data

# use the pca function defined in the assignment 3
def pca(data):
    """input a data matrix in which each column corresponds to a coordinate and each row corresponds to a data point;
    output the eigenvalues in a vector (numpy array) in descending order
    and a matrix where each column is an eigenvector of the corresponding eigenvalue
    """

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


print("begin PCA on the diatoms data")
# compute the needed values for plotting
variances, eigenvectors = pca(diatoms_data)
m = np.mean(diatoms_data, 0)

def plot_5_cells_givenPC(PC_index, variances, eigenvectors, m, col):
    """input a selected PC index (i.e. 1 or 2 or 3), the numpy array of variance and eigenvector matrix
    obtained from PCA, the mean of each column of the diatom data and a type of colour (e.g. "Greens", "Blues");
    generate the plot of 5 cells corresponding to this PC
    """

    # prepare the 3 variables
    theta_single = np.sqrt(variances[PC_index - 1])
    theta = numpy.matlib.repmat(theta_single, 180, 1).T[0]
    e = eigenvectors[:, PC_index - 1]

    # generate the 5 cells
    cell1 = m - 2 * theta * e
    cell2 = m - theta * e
    cell3 = m
    cell4 = m + theta * e
    cell5 = m + 2 * theta * e
    cell_list = [cell1, cell2, cell3, cell4, cell5]
    col_index_list = [0.5, 0.6, 0.7, 0.8, 0.9]

    # plotting
    colours = plt.get_cmap(col)
    for cell, ind in zip(cell_list, col_index_list):
        plot_a_cell(cell, colours(ind))


print("begin plotting the 5 cells for each PC")
# plot the 5 cells for each of the first 3 PCs
plt.cla()
plt.title("5 Cells for PC1")
plot_5_cells_givenPC(1, variances, eigenvectors, m, "Reds")
plt.savefig("5 Cells for PC1.png", format="png")

plt.cla()
plt.title("5 Cells for PC2")
plot_5_cells_givenPC(2, variances, eigenvectors, m, "Blues")
plt.savefig("5 Cells for PC2.png", format="png")

plt.cla()
plt.title("5 Cells for PC3")
plot_5_cells_givenPC(3, variances, eigenvectors, m, "Greens")
plt.savefig("5 Cells for PC3.png", format="png")
print("diatoms data finished")

### Exercise 3. Critical thinking

# use the mds function in the assignment 3
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


print("begin analyzing the PCA toy data")
# load the PCA toy data set and perform PCA
PCA_toy_data = np.loadtxt("pca_toydata.txt")
PCA_toy_data_projected = mds(PCA_toy_data, 2)
plt.cla()
plt.title("the PCA toy data projected on to the top 2 PCs")
plt.scatter(PCA_toy_data_projected[:, 0], PCA_toy_data_projected[:, 1])
plt.xlim(-1.75, 1.75)
plt.ylim(-1.5, 1.5)
plt.savefig("the PCA toy data projected on to the top 2 PCs.png", format="png")

# remove the last 2 data set and perform PCA again
PCA_toy_data_trim = PCA_toy_data[:-2, :]
PCA_toy_data_trim_projected = mds(PCA_toy_data_trim, 2)
plt.cla()
plt.title("the PCA toy data projected on to the top 2 PCs \n(without the last 2 data points)")
plt.scatter(PCA_toy_data_trim_projected[:, 0], PCA_toy_data_trim_projected[:, 1])
plt.xlim(-1.75, 1.75)
plt.ylim(-1.5, 1.5)
plt.savefig("the PCA toy data projected on to the top 2 PCs (without the last 2 data points).png", format="png")
print("the PCA toy data finished")

### Exercise 4. Clustering II

print("begin analyzing the pesticide data")
# load the pesticide data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')

# split the data into variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]

# project the data on the first 2 PCs
XTrain_projected = mds(XTrain, 2)

# collect the line index of weed points and crop points respectively
weed_line = []
crop_line = []
for i in range(len(YTrain)):
    if YTrain[i] == 0:  # weed
        weed_line.append(i)
    elif YTrain[i] == 1:  # crop
        crop_line.append(i)

# separate projected data points into 2 sets corresponding to the 2 classes
weed_point = XTrain_projected[weed_line, :]
crop_point = XTrain_projected[crop_line, :]

# plot the projected data points. weed points in yellow, crop points in green
plt.cla()
plt.axis('auto')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(weed_point[:, 0], weed_point[:, 1], color="yellow", label="weed")
plt.scatter(crop_point[:, 0], crop_point[:, 1], color="green", label="crop")

# perform k-means clustering
startingPoint = np.vstack((XTrain[0, ], XTrain[1, ]))
kmeans = KMeans(n_clusters=2, n_init=1, init=startingPoint, algorithm='full').fit(XTrain)

# center the coordinates of the 2 cluster centres (because all data points are centred during PCA)
XTrain_mean = np.mean(XTrain, 0)
XTrain_mean_matrix = numpy.matlib.repmat(XTrain_mean, kmeans.cluster_centers_.shape[0], 1)
cluster_centres_centered = kmeans.cluster_centers_ - XTrain_mean_matrix

# project the 2 cluster centres on the first 2 PCs obtaining from the whole training set
_, PCs = pca(XTrain)
PCs = PCs[:, 0:2]

cluster_centres_projected = np.dot(PCs.T, cluster_centres_centered.T).T
print("the 2 class centres projected on the first 2 PCs are:\n", cluster_centres_projected)

# plot the 2 class centres and finish plotting
plt.scatter(cluster_centres_projected[:, 0], cluster_centres_projected[:, 1], color="red", label="cluster centre")
plt.title("the Pesticide Data and Cluster Centres Projected on the First 2 PCs")
plt.legend(loc='upper center')
plt.savefig("the Pesticide Data and Cluster Centres Projected on the First 2 PCs.png", format="png")
print("pesticide data finished")
