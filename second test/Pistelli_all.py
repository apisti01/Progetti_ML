import numpy as np
import matplotlib.pyplot as plt


'''LOADING THE DATA'''
# Load the training and test data
X_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/X_train.txt', delimiter=' ')
y_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/y_train.txt', delimiter=' ')
X_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/X_test.txt', delimiter=' ')
y_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/y_test.txt', delimiter=' ')

# Extract the first column of the test data
X_test_height = X_test[:, 0]

# Assign all test samples either the male (0) or female (1) label based on their height
y_pred_male = np.zeros_like(X_test_height)
y_pred_female = np.ones_like(X_test_height)

# Load the true labels of the test data
y_true = y_test

# Calculate the percentage of correct classification for both cases (male and female)
correct_male = np.sum(y_pred_male == y_true)
correct_female = np.sum(y_pred_female == y_true)
percentage_male = correct_male / len(y_true) * 100
percentage_female = correct_female / len(y_true) * 100

print(f"Percentage of correct classification for male: {percentage_male:.2f}%")
print(f"Percentage of correct classification for female: {percentage_female:.2f}%")

import matplotlib.pyplot as plt

def gaussian_pdf(x, mu, sigma):
    """Calculate the Gaussian probability density function value for a given x, mean, and standard deviation."""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)

# Load the training data for males and females separately
X_train_male = X_train[y_train == 0, :]
X_train_female = X_train[y_train == 1, :]

# Calculate the mean and standard deviation of the height for males and females separately
mu_male, sigma_male = np.mean(X_train_male[:, 0]), np.std(X_train_male[:, 0])
mu_female, sigma_female = np.mean(X_train_female[:, 0]), np.std(X_train_female[:, 0])

# For each male and female training sample, calculate the Gaussian probability density function value using the corresponding mean and standard deviation
pdf_male = gaussian_pdf(X_train_male[:, 0], mu_male, sigma_male)
pdf_female = gaussian_pdf(X_train_female[:, 0], mu_female, sigma_female)

# Calculate the prior probabilities for males and females
prior_male = len(X_train_male) / len(X_train)
prior_female = len(X_train_female) / len(X_train)

# Calculate the likelihoods for all test samples using the corresponding mean and standard deviation
pdf_test_male = gaussian_pdf(X_test_height, mu_male, sigma_male)
pdf_test_female = gaussian_pdf(X_test_height, mu_female, sigma_female)

# Calculate the posterior probabilities for males and females using Bayes' theorem
posterior_male = pdf_test_male * prior_male
posterior_female = pdf_test_female * prior_female

# Classify the test samples based on the posterior probabilities
y_pred = np.zeros_like(y_true)
y_pred[posterior_female > posterior_male] = 1

# Calculate the percentage of correct classification
correct = np.sum(y_pred == y_true)
percentage = correct / len(y_true) * 100

print(f"Percentage of correct classification: {percentage:.2f}%")

# Plot the class-specific likelihoods for all training male and female samples using black circles for males and red circles for females
plt.scatter(X_train_male[:, 0], pdf_male, color='black', label='Male')
plt.scatter(X_train_female[:, 0], pdf_female, color='red', label='Female')
plt.xlabel('Height')
plt.ylabel('Likelihood')
plt.legend()
plt.show()


