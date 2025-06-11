# 🧠 Fashion MNIST Image Classification with Neural Networks

This project demonstrates a **basic deep learning pipeline** to classify images of clothing using the **Fashion MNIST dataset**. The implementation uses **TensorFlow** and **Keras**, and walks through the typical stages of loading data, preprocessing, model building, training, evaluation, and predictions.

---

## 📁 Dataset: Fashion MNIST

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) contains **70,000 grayscale images** of 28x28 pixels each, categorized into 10 fashion categories:

```
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

* **Training set**: 60,000 images
* **Test set**: 10,000 images

---

## 🧰 Libraries Used

| Library      | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `tensorflow` | Building, training, and evaluating neural networks |
| `keras`      | High-level API for defining and training models    |
| `numpy`      | Numerical operations on image arrays               |
| `matplotlib` | Visualizing sample images and model predictions    |

---

## 🚀 Project Workflow

### 1. **Import Libraries**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

### 2. **Load Dataset**

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

### 3. **Preprocessing**

* Normalize pixel values to range `[0, 1]` by dividing by 255.
* Visualize sample images using `matplotlib`.

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

### 4. **Build the Neural Network**

Using Keras Sequential API:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
```

### 5. **Compile the Model**

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### 6. **Train the Model**

```python
model.fit(train_images, train_labels, epochs=10)
```

### 7. **Evaluate the Model**

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

### 8. **Make Predictions**

```python
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
```

---

## 📊 Key Insights

* **Normalization** improves model training speed and stability.
* The **simple 2-layer neural network** (with ReLU and Softmax) achieves over **85% accuracy**.
* Misclassified images can be visualized to understand where the model struggles (e.g., Shirt vs T-shirt).
* This project sets the foundation for deeper experimentation like:

  * Adding Convolutional layers (CNNs)
  * Using Dropout to reduce overfitting
  * Hyperparameter tuning

---

## 🖼️ Sample Visualization

* First 25 training images with their class labels
* Model predictions on test samples

```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```

---

## 📦 Requirements

Install the following libraries before running the notebook:

```bash
pip install tensorflow matplotlib numpy
```

---

## 📘 Reference

* TensorFlow Tutorials: [Basic Classification](https://www.tensorflow.org/tutorials/keras/classification)
* Dataset: [Zalando Research - Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
