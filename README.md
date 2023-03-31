# Fashion MNIST Classification
Small school project to do classification using maching learning on fashion mnist dataset.

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

