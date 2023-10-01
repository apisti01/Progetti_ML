import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import urllib3
import pickle
import os

# load the single batch of CIFAR-10 data
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="bytes")
    return dict[b'data'], dict[b'labels']

# load all the data from the CIFAR-10 dataset
def load_full_data(dir):
    #cicle on the 5 batches of the dataset
    train_data = []
    train_labels = []
    for i in range(1, 6):
        file_path = os.path.join(dir, f'data_batch_{i}')
        data, labels = unpickle(file_path)
        train_data.append(data)
        train_labels.extend(labels)
    train_data = np.vstack(train_data)
    train_labels = np.array(train_labels)

    # Load test data
    test_data, test_labels = unpickle(os.path.join(dir, 'test_batch'))
    return (train_data, train_labels), (test_data, test_labels)


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = load_full_data('/home/apisti01/Erasmus/Progetti_ML/excercise_5/cifar-10-batches-py/')

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

# Create the neural network modelfi
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3072,)),  # Flatten the input
    keras.layers.Dense(20, activation='sigmoid'),      # Hidden layer with 20 neurons, no more 5
    keras.layers.Dense(20, activation='sigmoid'),      # second  Hidden layer with 20 neurons, no more 5
    keras.layers.Dense(10, activation='sigmoid')   # Output layer with 10 neurons (for 10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#print a summary of the model created
model.summary()

# Set the number of epochs and batch size
epochs = 40 # after 40 epochs the accuracy doesnt' increase anymore (tryed even with 300 epochs)
batch_size = 32

# Train the model
history = model.fit(x_train, y_train_one_hot, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(x_test,y_test_one_hot), 
                    verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=1)
print(f'Test accuracy: {test_accuracy:.4f}')

# Plot the training loss curve
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.show()