# Fashion MNIST Classification
Project to do classification using maching learning on fashion mnist dataset.
Using a simple neural network model, the result shows 87% accuracy in predicting the dataset.

[Update]
Added 2 more model and approach which perform slightly better.
1. A slightly more complex neural network along with validation data. Accuracy 90%
2. A 2 branch neural network model. Accuracy 90%


The code can be find in [classification.ipynb ](https://github.com/andricoc/fashion-mnist-classification/blob/main/classification.ipynb)

## Overview
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."

Zalando seeks to replace the original MNIST dataset

Original dataset from : https://github.com/zalandoresearch/fashion-mnist
![image](https://user-images.githubusercontent.com/63791918/229101992-b1e9818c-7020-4289-9e06-b7e797bf0f09.png)


## Method
The process is quite straightforward. Only minor data preprocessing are done.
```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
```

After that, we fit the data into the model. This time i am using a simple custom neural network model.

![image](https://user-images.githubusercontent.com/63791918/229102273-8716a758-47cd-469d-b990-56af53abb4dd.png)

## Result
![image](https://user-images.githubusercontent.com/63791918/229102538-4a946acf-ef10-4765-a24a-e42e67fa40fd.png)

### Visual representation
![image](https://user-images.githubusercontent.com/63791918/229102679-de5a996d-f4a2-4b6e-b46d-4e92c01871a9.png)

# Improvement

## Slightly Complex Model With Validation Data

Split the data with 80 : 20 ratio for train and validation
```python
# Split the train data to train and validation data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2019)
```
The model:

![image](https://github.com/ricocahyadi777/fashion-mnist-classification/assets/63791918/cc39cf94-c200-4bb3-aea3-0d13f001ea62)

Result:

![image](https://github.com/ricocahyadi777/fashion-mnist-classification/assets/63791918/f8c27ad6-6395-4326-93f3-bb37fd42a59e)

## Two-Branch Model

The model:

```python
# Make the two branch model
inputs = tf.keras.Input(shape = (28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
block_1 = layers.Flatten()(x)

x = layers.Flatten()(inputs)
block_2 = layers.Dense(64, activation = 'relu')(x)
                                     
converge = tf.keras.layers.concatenate([block_1, block_2])

x = layers.Dense(32, activation='relu') (converge)
outputs = layers.Dense(10, activation='softmax') (x)
                                     
model2 = tf.keras.Model(inputs, outputs)
```

![image](https://github.com/ricocahyadi777/fashion-mnist-classification/assets/63791918/84e585aa-3453-4341-bc7e-ba6322d835e1)

Result:

![image](https://github.com/ricocahyadi777/fashion-mnist-classification/assets/63791918/68f962d3-d476-409d-9535-61608af4b177)
