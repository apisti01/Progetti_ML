import numpy as np


#Import the test data from file
test_data = np.loadtxt('excercise_4/male_female_y_test.txt', delimiter= ' ', dtype=float)

#randomly assign a value for every sample
random = np.random.randint(0, 2, len(test_data))

#compute the accuracy
random_accuracy = np.sum(random == test_data) / len(test_data)

print('The accuracy is: ', random_accuracy)

train_datas = np.loadtxt('excercise_4/male_female_y_train.txt', delimiter= ' ', dtype=float).astype(int)

#calculate the a priori probabilities
counts = np.bincount(train_datas)
apriori = counts / len(train_datas)
prediction = np.argmax(apriori)

#compute the accuracy
apriori_accuracy = np.sum(prediction == test_data) / len(test_data)

print('The accuracy is: ', apriori_accuracy)