---
layout: page
title: Teaching Session 2
description: >-
    Teaching Session 2 Materials and information
---

# TNM112 -- Teaching Session 2 
{:.no_toc}

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction to Keras

In this notebook, we will train some machine learning models using Keras framework.
We will also use Scikit-learn for data preprocessing and evaluation. Kindly check the
following links for more details:

1. https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
2. https://keras.io/getting_started/
3. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow
import sqlite3

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from pandas.plotting import parallel_coordinates
```

# 1. IRIS Flowers Classifier

In this section, we will train a neural network to classify IRIS flowers dataset. Here, we’ll walk through the implementation steps so you can use this information to train another classifier in the next section. 

## 1.1. Load the dataset

For the dataset, we use IRIS dataset. IRIS dataset consist of 150 datapoints with four input features ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"] and three output classes ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]. You can [download](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) the iris flowers dataset from the UCI Machine Learning repository.

In the cell below, the IRIS dataset is loaded using [Pandas](https://pandas.pydata.org/) and visualized using parallel-plot


```python
# Load data from URL using Pandas
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
col_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
data =  pd.read_csv(csv_url, names = col_names)

#Do Parallel Plot
parallel_coordinates(data, 'Species', color=('#FF0000', '#0000FF', '#FFFF00'))
plt.figure()

# Display information about the dataset
print('=====================================================================================================')
print('First five rows in the dataset:\n', data.head(5))
print('=====================================================================================================')
print('Information about Data:')
print(data.info())

# Split dataset into input features and class labels
Y = data['Species']
X = data.drop(['Species'], axis=1)
print('=====================================================================================================')
print("Shape of Input  features: {}".format(X.shape))
print("Shape of Output features: {}".format(Y.shape))
print('=====================================================================================================')
print("Check the number of datapoints for each class label:")
print(Y.value_counts())

#One hot encode the class labels
lbl_clf = LabelEncoder()
Y_encoded = lbl_clf.fit_transform(Y)
Y_final = tensorflow.keras.utils.to_categorical(Y_encoded)
print('=====================================================================================================')
print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))
```

## 1.2. Data Splitting and Normalization

From the dataset details, you can see that the IRIS dataset contains 150 samples. In the cell below, the data is split into two sets, with 75% of the samples for the training set and the remaining 25% for the test set.

Next, the input features are normalized using `StandardScaler` from scikit-learn. Normalization adjusts the feature values to have a mean of zero and a standard deviation of one, helping to improve the model's performance and training stability.


```python
seed=42

#Split the dataset into train and test set using train_test_split() from sklearn
x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.25, random_state=seed, stratify=Y_encoded, shuffle=True)

#Normalize the dataset using StandardScaler() from sklearn
std_clf = StandardScaler()
x_train_std = std_clf.fit_transform(x_train)
x_test_std = std_clf.transform(x_test)

print("Training Input shape\t: {}".format(x_train_std.shape))
print("Testing Input shape\t: {}".format(x_test_std.shape))
print("Training Output shape\t: {}".format(y_train.shape))
print("Testing Output shape\t: {}".format(y_test.shape))
```

## 1.3. Defining the Neural Network Architecture

In the cell below, a neural network is defined using Keras. This architecture includes two hidden layers and is specifically designed to classify the IRIS dataset.

**Network Architecture:**

   1. **`First Hidden Layer:`** This layer has 10 neurons with ReLU activation, taking in the 4-dimensional input. Kernel is initialized with Normal distribution and uses L2 regularization (l2=0.01) to help reduce overfitting.

   2. **`Batch Normalization:`** Applied after the first hidden layer to standardize its output.

   3. **`Dropout Layer:`** 30% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.

   4. **`Second Hidden Layer:`** This layer has 5 neurons, also using ReLU activation.  Kernel is initialized with Normal distribution and uses L2 regularization (l2=0.01) to help reduce overfitting. 

   5. **`Batch Normalization:`** Applied after the second hidden layer to standardize its output.
   
   6. **`Dropout Layer:`** 30% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.
   
   7. **`Output Layer:`** This layer has 3 neurons, one for each class in the IRIS dataset, and uses Softmax activation to produce a probability distribution across the classes.


```python
#Define the neural network architecture. Check Keras documentation for more info
import tensorflow
from tensorflow import keras

#Define a Sequential model
model = keras.models.Sequential(name="MLP-1")

#First Hidden Layer with 10 neurons that takes 4 dimensional input value and relu activation
model.add(keras.layers.Dense(10, input_dim=4, activation=tensorflow.nn.relu, kernel_initializer="normal",
                                kernel_regularizer=keras.regularizers.l2(0.01),
                                name="hidden_layer_1"))

#Apply Batch Normalization to the output values of the first hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_1"))

#Adding Dropout to the first hidden layer with probability of 0.3
model.add(keras.layers.Dropout(0.3,name="dropout_1"))

#Second Hidden Layer with 5 neurons that takes 10 dimensional input value from previous layer and relu activation
model.add(keras.layers.Dense(5, activation = tensorflow.nn.relu, kernel_initializer="normal",
                                kernel_regularizer=keras.regularizers.l2(0.01),
                                name="hidden_layer_2"))

#Apply Batch Normalization to the output values of the second hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_2"))

#Adding Dropout to the second hidden layer with probability of 0.3
model.add(keras.layers.Dropout(0.3, name="dropout_2"))

#Output Layer with Softmax activation
model.add(keras.layers.Dense(3, activation=tensorflow.nn.softmax,name="output_layer"))

#Once a model is "built", you can call its summary() method to display its contents
model.summary()
```

## 1.4. Configuring and Training the Model

In the cell below, the model will be trained based on the following hyperparameters.

- **Optimizer:**  Adam optimizer

- **Loss Function:** categorical_crossentropy

- **Epochs:** 5

- **Batch Size:** 7

- **Metrics:** Accuracy


```python
#Set seed
tensorflow.random.set_seed(42)

#Configure the model for training. Define the hyperparatmeters: optimizer, loss and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
iris_model = model.fit(x_train_std, y_train, epochs=5, batch_size=7)
```

## 1.5. Evaluation on Test set
In this cell, the trained model is evaluated on the test set and analyze its performance using key classification metrics.


```python
from sklearn.metrics import classification_report

#Evaluate the model on test set
score = model.evaluate(x_test_std, y_test, verbose=0)

#Score has two values. The first value represent loss and the second value represent the accuracy
print("Test loss:      ", score[0])
print("Test accuracy:  ", 100*score[1])

#Get the model predictions on test set
y_pred = model.predict(x_test_std)
#Get the index of the highest value for each predictions (predicted class labels)
y_pred = np.argmax(y_pred, axis = 1)
#Convert the one hot vector to True class labels
y_test_oh = np.argmax(y_test, axis =1)

#Compute Precision, Recall, F1-Score and Accuracy of the model on the test set
print('=====================================================================================================')
print(classification_report(y_test_oh, y_pred, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]))
print('=====================================================================================================')
```

## 2. Penguin Species Classifier

In this section, we will train a neural network to classify three penguin species based on the features: `bill length`, `bill depth`, `flipper length` and `body mass`. Here, you will do the step-by-step implementation of the classifier.


## Task 2.1: Loading data

For the dataset, we use Palmer Penguins dataset. The dataset consist of 344 datapoints with four input features : `bill length`, `bill depth`, `flipper length` and `body mass`. and three output classes `Adelie`, `Chinstrap` and `Gentoo`.

In the cell below, the Palmer Penguins dataset is loaded as a DataFrame. You will visualize the dataset using parallel-plot.


```python
import seaborn as sns

data = sns.load_dataset("penguins")

# Drop "NaN" values and column with text
data = data.dropna()
data = data.drop(columns=['island', 'sex'])

# Split dataset into input features and class labels
Y = data['species']
X = data.drop(['species'], axis=1)

#One hot encode the class labels
lbl_clf = LabelEncoder()
Y_encoded = lbl_clf.fit_transform(Y)
Y_final = tensorflow.keras.utils.to_categorical(Y_encoded)

print(f"Length of the dataset:{len(data)}\n")
print(data.head())

# Task 1: Do a parallel-plot of the data
parallel_coordinates(data, 'species', color=('#FF0000', '#0000FF', '#FFFF00'))
plt.figure()
```

## Task 2.2: Data Splitting and Normalization

From the dataset details, you can see that the Penguins dataset contains 333 samples (after removing "NaNs"). 

In the cell below, you will split the data, using 80% of the samples for the training set and the remaining 20% for the test set. Next, you will normalize the features using `StandardScaler` from scikit-learn.


```python
seed=42

# Step 1: Split the dataset into train and test set using train_test_split() from sklearn
x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.20, random_state=seed, stratify=Y_encoded, shuffle=True)

# Step 2: Normalize the dataset using StandardScaler() from sklearn
std_clf = StandardScaler()
x_train_std = std_clf.fit_transform(x_train)
x_test_std = std_clf.transform(x_test)

print("Training Input shape\t: {}".format(x_train_std.shape))
print("Testing Input shape\t: {}".format(x_test_std.shape))
print("Training Output shape\t: {}".format(y_train.shape))
print("Testing Output shape\t: {}".format(y_test.shape))
```

## Task 2.3: Defining the Neural Network Architecture

In the cell below, you will define a neural network model using Keras. This architecture includes 3 hidden layers and is specifically designed to classify the IRIS dataset.

Design the network based on the following information:

   1. **`First Hidden Layer:`** This layer has 10 neurons with ReLU activation, taking in the 4-dimensional input. Kernel is initialized with He_Normal distribution and uses L2 regularization (l2=0.01) to help reduce overfitting.

   2. **`Batch Normalization:`** Applied after the first hidden layer to standardize its output.

   3. **`Dropout Layer:`** 25% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.

   4. **`Second Hidden Layer:`** This layer has 10 neurons, also using ReLU activation.  Kernel is initialized with He_Normal distribution and uses L1L2 regularization (experiment with L1 and L2 values) to help reduce overfitting. 

   5. **`Batch Normalization:`** Applied after the second hidden layer to standardize its output.
   
   6. **`Dropout Layer:`** 25% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.
   
   7. **`Third Hidden Layer:`** This layer has 8 neurons, also using ReLU activation.  Kernel is initialized with He_Normal distribution and uses L1L2 regularization (experiment with L1 and L2 values) to help reduce overfitting. 

   8. **`Batch Normalization:`** Applied after the third hidden layer to standardize its output.
   
   9. **`Dropout Layer:`** 25% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.
   
   10. **`Fourth Hidden Layer:`** This layer has 8 neurons with ReLU activation, taking in the 4-dimensional input. Kernel is initialized with He_Normal distribution and uses L2 regularization (l2=0.01) to help reduce overfitting.

   11. **`Batch Normalization:`** Applied after the fourth hidden layer to standardize its output.

   12. **`Dropout Layer:`** 25% of the neurons are randomly dropped during training, which adds robustness by reducing reliance on specific neurons.
   
   13. **`Output Layer:`** This layer has 3 neurons, one for each class in the Penguins dataset, and uses Softmax activation to produce a probability distribution across the classes.


```python
# Import necessary libraries
import tensorflow
from tensorflow import keras

# Define the Sequential model for the Penguin dataset
model = keras.models.Sequential(name="Penguin_Classifier")

# First Hidden Layer with 10 neurons that takes 4-dimensional input and ReLU activation
model.add(keras.layers.Dense(10, input_dim=4, activation=tensorflow.nn.relu, 
                             kernel_initializer="he_normal", 
                             kernel_regularizer=keras.regularizers.l2(0.01), 
                             name="hidden_layer_1"))

# Apply Batch Normalization to the output values of the first hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_1"))

# Adding Dropout to the first hidden layer with a dropout rate of 0.25
model.add(keras.layers.Dropout(0.25, name="dropout_1"))

# Second Hidden Layer with 10 neurons, ReLU activation, and L1L2 regularization
model.add(keras.layers.Dense(10, activation=tensorflow.nn.relu, 
                             kernel_initializer="he_normal", 
                             kernel_regularizer=keras.regularizers.L1L2(), 
                             name="hidden_layer_2"))

# Apply Batch Normalization to the output values of the second hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_2"))

# Adding Dropout to the second hidden layer with a dropout rate of 0.25
model.add(keras.layers.Dropout(0.25, name="dropout_2"))

# Third Hidden Layer with 8 neurons, ReLU activation, and L1L2 regularization
model.add(keras.layers.Dense(8, activation=tensorflow.nn.relu, 
                             kernel_initializer="he_normal", 
                             kernel_regularizer=keras.regularizers.L1L2(), 
                             name="hidden_layer_3"))

# Apply Batch Normalization to the output values of the third hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_3"))

# Adding Dropout to the third hidden layer with a dropout rate of 0.25
model.add(keras.layers.Dropout(0.25, name="dropout_3"))

# Fourth Hidden Layer with 8 neurons, ReLU activation, and L2 regularization
model.add(keras.layers.Dense(8, activation=tensorflow.nn.relu, 
                             kernel_initializer="he_normal", 
                             kernel_regularizer=keras.regularizers.l2(0.01), 
                             name="hidden_layer_4"))

# Apply Batch Normalization to the output values of the fourth hidden layer
model.add(keras.layers.BatchNormalization(name="batchnorm_4"))

# Adding Dropout to the fourth hidden layer with a dropout rate of 0.25
model.add(keras.layers.Dropout(0.25, name="dropout_4"))

# Output Layer with 3 neurons (one for each penguin species) and Softmax activation
model.add(keras.layers.Dense(3, activation=tensorflow.nn.softmax, name="output_layer"))

# Display the model summary
model.summary()

```

## Task 2.4: Configuring and Training the Model

In the cell below, we will configure the neural network model for training based on the following hyperparameters.

- **Optimizer:**  AdamW optimizer

- **Loss Function:** categorical_crossentropy

- **Epochs:** 20

- **Batch Size:** 16

- **Metrics:** F1 Score


```python
#Set seed
tensorflow.random.set_seed(42)

#Configure the model for training. Define the hyperparatmeters: optimizer, loss and metrics
model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['f1_score'])

#Train the model
penguin_model = model.fit(x_train_std, y_train, epochs=20, batch_size=16)
```

## Task 2.5: Plot Loss Curve

Visualize the training loss from the above training process using `matplotlib`. The training data can be accessed through `penguin_model.history`.


```python
penguin_model.history['loss']
```


```python
import matplotlib.pyplot as plt

# Assuming 'penguin_model' is the result from model.fit()
history = penguin_model.history

# Plotting the loss
plt.plot(history['loss'], label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

```

## Task 2.6: Evaluation on Test set

In this cell, you will find the `Precision`, `Recall` and `F1-score` of the model on the test set.


```python
from sklearn.metrics import classification_report

#Get the model predictions on test set
y_pred = model.predict(x_test_std)
#Get the index of the highest value for each predictions (predicted class labels)
y_pred = np.argmax(y_pred, axis = 1)
#Convert the one hot vector to True class labels
y_test_oh = np.argmax(y_test, axis =1)

#Compute Precision, Recall, F1-Score and Accuracy of the model on the test set
print('=====================================================================================================')
print(classification_report(y_test_oh, y_pred, target_names=["Adelie", "Chinstrap", "Gentoo"]))
print('=====================================================================================================')
```

# 3. Live Plot and Predictions 

In this task, we will visualize the train and test accuracy along with the test set predictions after every batch in an epoch using live plot.

## 3.1. Helper Function for Plotting


```python
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output
import random

def imgrid(x,y,yp,xx,yy):
    ind = [i for i in range(x.shape[0])]
    random.shuffle(ind)

    plt.figure(figsize=(18,yy*2))
    for i in range(xx*yy):
        plt.subplot(yy,xx,i+1)
        if x.shape[3]==1:
            plt.imshow(x[ind[i],:,:,0],cmap='gray')
        else:
            plt.imshow(x[ind[i],:,:,:])

        if len(yp)>0:
            plt.title('p=%d, gt=%d'%(yp[ind[i]],y[ind[i]]))
        else:
            plt.title('label=%d'%(y[ind[i]]))
        plt.axis('off')
    plt.show()

def live_plot(x, y, yp, acc, acc_test, batch, bs, N, xx, yy):
    clear_output(wait=True)

    ind = [i for i in range(x.shape[0])]

    fig = plt.figure(figsize=(18, 10))
    outer = gridspec.GridSpec(2, 1, hspace=0.2)
    inner = gridspec.GridSpecFromSubplotSpec(yy, xx,
                    subplot_spec=outer[0], wspace=0.1, hspace=0.0)

    for i in range(xx*yy):
        ax = plt.Subplot(fig, inner[i])
        if x.shape[3]==1:
            ax.imshow(x[ind[i],:,:,0],cmap='gray')
        else:
            ax.imshow(x[ind[i],:,:,:])
        if yp[ind[i]] == y[ind[i]]:
            ax.set_title('Pred = %d'%(yp[ind[i]]), color='g')
        else:
            ax.set_title('Pred = %d'%(yp[ind[i]]), color='r')
        ax.axis('off')
        fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[1], wspace=0.0, hspace=0.1)
    ax = plt.Subplot(fig, inner[0])
    ax.plot(np.linspace(0,batch*bs/N,len(acc)),100.0*np.array(acc),label='Training')
    ax.plot(np.linspace(0,batch*bs/N,len(acc_test)),100.0*np.array(acc_test),label='Test')
    ax.plot(batch*bs/N,100*acc[-1],'o')
    ax.plot(batch*bs/N,100*acc_test[-1],'o')
    ax.legend()
    ax.grid(1)
    ax.set_xlim([0,np.maximum(1,batch*bs/N)])
    ax.set_ylim([np.minimum(np.min(100.0*np.array(acc)),np.min(100.0*np.array(acc_test))),100])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    fig.add_subplot(ax)

    plt.show()

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.l_train = []
        self.l_test = []
        self.bs = 128
        self.batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.l_train.append(logs['accuracy'])
        self.batch += 1

        if np.mod(batch,10)==0:
            score = model.evaluate(x_test, y_test, verbose=0)
            self.l_test.append(score[1])

            yp_test = np.argmax(model.predict(x_test[:24]),1)
            live_plot(x_test,np.argmax(y_test[:24],1),yp_test,self.l_train,self.l_test,self.batch,self.bs,len(x_train)-self.bs,12,2)
```

## 3.2. Load MNIST Data

The MNIST dataset is a collection of handwritten digits, consisting of 60,000 training images and 10,000 test images, each 28x28 pixels in grayscale. It is commonly used for training and evaluating machine learning models, particularly for image classification tasks.


```python
import numpy as np
import keras

# load MNIST dataset from keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One hot encoding of class labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Display images
imgrid(x_train,np.argmax(y_train,1),[],12,3)
```

## 3.3. Model Definition and Training

We will define Convolutional Neural Networks for this task, which will be covered in the lab 2 and the next teaching session.


```python
import tensorflow
from tensorflow import keras
from keras import layers

keras.backend.clear_session()

#Define model architecture
model = keras.Sequential(
      [
          layers.InputLayer(input_shape=(28,28,1)),
          layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
          layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
          layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
          layers.MaxPooling2D(pool_size=(2, 2)),
          layers.Flatten(),
          layers.Dense(128, activation="relu"),
          layers.Dense(10, activation="softmax"),
      ])

model.summary()
```


```python
opt = keras.optimizers.Adam()
#Configure the model for training
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#Train the model for 1 Epoch
model.fit(x_train, y_train, batch_size=128, epochs=1,
          callbacks=[CustomCallback()],
          validation_split=0.0, verbose=0);
```

## 3.4. Evaluation


```python
#Displaying the training performance
score = model.evaluate(x_train, y_train, verbose=0)
print("Train loss:     ", score[0])
print("Train accuracy: ", 100*score[1])

#Displaying the test performance
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:      ", score[0])
print("Test accuracy:  ", 100*score[1])
```