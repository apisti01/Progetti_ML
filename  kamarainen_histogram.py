import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/male_female_X_train.txt', delimiter=' ', dtype=float)
grtr = np.loadtxt('/home/apisti01/Erasmus/Progetti_ML/male_female_y_train.txt', delimiter=' ', dtype=float)

#import from a file data, the fisrt column is the heights and the second column is the weights
heights = data[:,0]
weights = data[:,1]

tmp = [data[:,0], grtr]

# Create the histogram
plt.hist(tmp, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Histogram of two vectors')
plt.show()