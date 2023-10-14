import numpy as np
import matplotlib.pyplot as plt

'''LOADING THE DATA'''
# Load the training and test data
X_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/X_train.txt', delimiter=' ')
y_train = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/y_train.txt', delimiter=' ')
X_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/X_test.txt', delimiter=' ')
y_test = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/second test/y_test.txt', delimiter=' ')

# Filter the male and female heights from the training data based on the class labels
male_heights = X_train[y_train == 0, 0]
female_heights = X_train[y_train == 1, 0]

# Plot the histograms of the male and female heights on the same plot
plt.hist(male_heights, alpha=0.5, label='Male')
plt.hist(female_heights, alpha=0.5, label='Female')

# Add a legend to the plot to show the class labels
plt.legend()

# Name the y- and x-axes correctly
plt.ylabel('Frequency')
plt.xlabel('Height')

# Show the plot
plt.show()

