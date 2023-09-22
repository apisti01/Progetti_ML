import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_data = np.loadtxt('excercise_4/male_female_X_train.txt', delimiter=' ', dtype=float)
test_data = np.loadtxt('excercise_4/male_female_X_test.txt', delimiter=' ', dtype=float)
control_data_train = np.loadtxt('excercise_4/male_female_y_train.txt', delimiter=' ', dtype=float).astype(int)
control_data_test = np.loadtxt('excercise_4/male_female_y_test.txt', delimiter=' ', dtype=float).astype(int)


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

#a priori probabilities
counts = np.bincount(control_data_train)
apriori = counts / len(control_data_train)
print('The a priori probabilities are (male and female): ', apriori)

# Classify test samples
def classify_sample_hw(height, weight):
    # Find the bin index for height and weight
    height_bin_index = np.digitize(height, male_height_bins)
    weight_bin_index = np.digitize(weight, male_weight_bins)

    # Compute likelihoods for male and female
    male_height_likelihood = male_height_hist[height_bin_index - 1] / len(male_train_data[:, 0])
    male_weight_likelihood = male_weight_hist[weight_bin_index - 1] / len(male_train_data[:, 1])
    female_height_likelihood = female_height_hist[height_bin_index - 1] / len(female_train_data[:, 0])
    female_weight_likelihood = female_weight_hist[weight_bin_index - 1] / len(female_train_data[:, 1])

    # Calculate posterior probabilities
    posterior_male = male_height_likelihood * male_weight_likelihood
    posterior_female = female_height_likelihood * female_weight_likelihood

    # Classify the sample
    if posterior_male > posterior_female:
        return 0
    else:
        return 1


def classify_sample_h(height):
     # Find the bin index for height
    height_bin_index = np.digitize(height, male_height_bins)

    # Compute likelihoods for male and female
    male_height_likelihood = male_height_hist[height_bin_index - 1] / len(male_train_data[:, 0])
    female_height_likelihood = female_height_hist[height_bin_index - 1] / len(female_train_data[:, 0])

    #classifiy the sample
    if male_height_likelihood > female_height_likelihood:
        return 0
    else:
        return 1
    
def classify_sample_w(weight):
    # Find the bin index for weight
    weight_bin_index = np.digitize(weight, male_weight_bins)

    #compute likelihoods for male and female
    male_weight_likelihood = male_weight_hist[weight_bin_index - 1] / len(male_train_data[:, 1])
    female_weight_likelihood = female_weight_hist[weight_bin_index - 1] / len(female_train_data[:, 1])

    #classify the sample
    if male_weight_likelihood > female_weight_likelihood:
        return 0    
    else:
        return 1

# Classify test samples and calculate accuracy
correct_height_only = 0
correct_weight_only = 0
correct_both = 0

for i in range(len(control_data_test)):
    if classify_sample_hw(test_data[i,0],test_data[i,1]) == control_data_test[i]:
        correct_both += 1
    if classify_sample_h(test_data[i,0]) == control_data_test[i]:
        correct_height_only += 1
    if classify_sample_w(test_data[i,1]) == control_data_test[i]:
        correct_weight_only += 1

accuracy_height_only = correct_height_only / len(control_data_test)
accuracy_weight_only = correct_weight_only / len(control_data_test)
accuracy_both = correct_both / len(control_data_test)

print(f"Accuracy (Height Only): {accuracy_height_only:.2%}")
print(f"Accuracy (Weight Only): {accuracy_weight_only:.2%}")
print(f"Accuracy (Both Height and Weight): {accuracy_both:.2%}")