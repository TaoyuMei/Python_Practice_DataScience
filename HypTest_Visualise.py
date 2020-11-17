import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

data = np.loadtxt('smoking.txt')

### Exercise 1. reading and processing data

# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers
def meanFEV1(data):
    """accept the data matrix and return the tuple of the average FEV1 of smokers and non-smokers"""
    non_smoker = np.mean([FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 0])
    smoker = np.mean([FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 1])
    return non_smoker, smoker


non_smoker, smoker = meanFEV1(data)
print("The mean FEV1 of non-smokers:", non_smoker, "\nThe mean FEV1 of smokers:", smoker)

### Exercise 2. draw boxplots of the FEV1 of non-smokers and smokers respectively

non_smoker_FEV1 = [FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 0]
smoker_FEV1 = [FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 1]
plt.boxplot([non_smoker_FEV1, smoker_FEV1], labels=["non-smokers", "smokers"])
plt.title("Boxplot over the FEV1 of Non-smokers and Smokers")
plt.ylabel("FEV1")
plt.savefig('Boxplot FEV1.png', format='png')

### Exercise 3. hypothesis testing

# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
def hyptest(data):
    """accept the data matrix, return True if the null hypothesis is rejected and False otherwise;
    although the np.var() function does not calculate the unbiased estimation of variance, the result is not affected,
    as the sample size is relatively large"""
    non_smoker_FEV1 = [FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 0]
    smoker_FEV1 = [FEV1 for _, FEV1, _, _, smoking_status, _ in data if smoking_status == 1]
    var_non_smoker = np.var(non_smoker_FEV1)
    var_smoker = np.var(smoker_FEV1)
    n_non_smoker = len(non_smoker_FEV1)
    n_smoker = len(smoker_FEV1)
    mean_non_smoker = np.mean(non_smoker_FEV1)
    mean_smoker = np.mean(smoker_FEV1)

    T = (mean_smoker - mean_non_smoker) / np.sqrt((var_non_smoker / n_non_smoker) + (var_smoker / n_smoker))
    print("The value of the t-statistic:", T)
    df_numerator = ((var_non_smoker / n_non_smoker) + (var_smoker / n_smoker)) ** 2
    df_denominator1 = var_non_smoker ** 2 / (n_non_smoker ** 2 * (n_non_smoker - 1))
    df_denominator2 = var_smoker ** 2 / (n_smoker ** 2 * (n_smoker - 1))
    df = df_numerator / (df_denominator1 + df_denominator2)
    print("The degree of freedom:", df)
    p = 2 * t.cdf(-T, df)
    print("The p-value:", p)
    return p < 0.05


if hyptest(data):
    print("The null hypothesis is rejected")
else:
    print("Cannot reject the null hypothesis")


### Exercise 4. correlation between age and FEV1

# calculate the correlation
correlation = np.cov(data, rowvar=False)[0, 1] / (np.sqrt(np.var(data[:, 1])) + np.sqrt(np.var(data[:, 0])))
# np.cov generate the covariance matrix; 'rowvar=False' means each column in the data corresponds to a variable;
# in the covariance matrix, [0, 1] is the covariance between FEV1 and age
print("The correlation between FEV1 and age:", correlation)
# plot the correlation
plt.cla()  # remove existing plots to avoid overlap
plt.scatter(data[:, 1], data[:, 0])  # data[:, 1] is FEV1, data[:, 0] is age
plt.xlabel("age")
plt.ylabel("FEV1")
plt.savefig("scatter plot FEV1_age.png", format='png')

### Exercise 5. histograms

non_smoker_age = [age for age, _, _, _, smoking_status, _ in data if smoking_status == 0]
smoker_age = [age for age, _, _, _, smoking_status, _ in data if smoking_status == 1]
plt.cla()  # remove existing plots to avoid overlap
plt.hist(non_smoker_age, label='non-smokers', edgecolor='black')
plt.hist(smoker_age, label='smokers', facecolor='red', edgecolor='black')
plt.xlabel("Age in year")
plt.ylabel("Count")
plt.title("Histograms over the Age of Subjects in Non-smoking and Smoking Groups")
plt.legend()
plt.savefig("histogram_age.png", format="png")
