import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_data = np.loadtxt('male_female_X_train.txt', delimiter=' ', dtype=float)
#test_data = np.loadtxt('test.txt', delimiter=',')
control_data_train = np.loadtxt('male_female_y_train.txt', delimiter=' ', dtype=float)


# Split the data into male and female data
male_train_data = train_data[control_data_train[:] == 0]
female_train_data = train_data[control_data_train[:] == 1]
#male_test_data = test_data[test_data[:, 0] == 1]
#female_test_data = test_data[test_data[:, 0] == 0]

# Compute the histograms
male_height_hist, male_height_bins = np.histogram(male_train_data[:, 0], bins=10, range=(80, 220))
female_height_hist, female_height_bins = np.histogram(female_train_data[:, 0], bins=10, range=(80, 220))
male_weight_hist, male_weight_bins = np.histogram(male_train_data[:, 1], bins=10, range=(30, 180))
female_weight_hist, female_weight_bins = np.histogram(female_train_data[:, 1], bins=10, range=(30, 180))

# Plot the histograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.hist(male_height_bins[:-1], bins= male_height_bins, weights=male_height_hist, label = 'male',alpha = 0.5)
plt.hist(female_height_bins[:-1],bins=female_height_bins, weights= female_height_hist,label='female', alpha = 0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.title('Height histogram')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist(male_weight_bins[:-1], bins = male_weight_bins, weights=male_weight_hist, label='Male', alpha = 0.5)
plt.hist(female_weight_bins[:-1], bins = female_weight_bins, weights=female_weight_hist, label='Female', alpha = 0.5)
plt.xlabel('Weight (kg)')
plt.ylabel('Frequency')
plt.title('Weight histogram')
plt.legend()

plt.tight_layout()
plt.show()

