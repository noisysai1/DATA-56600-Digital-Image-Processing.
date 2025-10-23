# Week 3 – Image Classification and Convolutional Neural Networks (CNN)

### 🧭 Overview
This week focused on applying **Convolutional Neural Networks (CNNs)** for image processing and classification tasks. 
The experiment used a sample image of the **Chicago Downtown skyline** and the **MNIST dataset** to explore how CNNs identify spatial features like edges, textures and object structures. 
The goal was to understand how neural architectures extract meaningful patterns and classify visual information effectively.

---

## 🧩 Key Concepts

### 1️⃣ Convolutional Neural Networks (CNN)
CNNs are a class of deep learning models highly effective for visual data analysis. 
They automatically extract and learn features such as edges, textures, and shapes through a series of convolutional, activation and pooling layers.

**Core Components:**
- **Convolutional Layer:** Applies filters (kernels) to extract spatial features such as edges or curves.
- **Activation Function:** Introduces non-linearity; ReLU (Rectified Linear Unit) is most commonly used.
- **Pooling Layer:** Reduces feature map dimensions (via max or average pooling) to minimize computation while retaining key information.
- **Fully Connected Layer:** Combines extracted features for final classification.

---

## ⚙️ Implementation and Algorithm Design

### 🔹 Dataset
- Used **MNIST dataset** for training and validation.
- Each image is of size **28x28 pixels**, grayscale.
- Split into training (60,000 images) and testing (10,000 images).

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

🔹 Data Preprocessing

Reshaped data depending on Keras backend configuration (channels_first or channels_last).

Normalized pixel values to range [0, 1] for faster convergence.

Converted labels into one-hot encoding for multi-class classification.

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)

🔹 Fully Connected Neural Network (FCNN)

As a baseline, a simple fully connected neural network (Dense layers only) was trained for comparison.

Architecture:

Flatten layer (28×28 → 784 neurons)

Two dense layers (512 neurons each, ReLU activation)

Output layer (10 neurons, Softmax activation)

Code Snippet:

from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=512, validation_split=0.1,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])


Performance:
✅ Validation Accuracy: ~97.6%


🔹 Convolutional Neural Network (CNN)

Next, a CNN was built to extract spatial hierarchies from the same dataset and perform image classification.

Architecture:

Conv2D(32, (5×5)) – Extracts basic features

MaxPooling2D(2×2) – Reduces dimensionality

Conv2D(64, (5×5)) – Captures complex features

MaxPooling2D(2×2) – Further reduction

Flatten → Dense(1024) with ReLU activation

Dense(10) with Softmax activation for final classification


Code Snippet:

from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Conv2D(32, (5,5), input_shape=(28,28,1)),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, (5,5)),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
history = model.fit(x_train, y_train, batch_size=64, epochs=512, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

📈 Model Evaluation and Results


🔹 CNN Model Accuracy
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
cnn_accuracy = accuracy_score(y_pred_classes, y_test)
print(f'Accuracy on hold-out set: {cnn_accuracy * 100:.2f}%')


✅ Accuracy on hold-out set: 99.00%

Significantly higher than the FCNN model (~97.6%)

Demonstrates CNN’s ability to learn spatial features more effectively.

🔹 Training and Validation Loss Plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()


This plot showed that loss decreased steadily before stabilizing, confirming efficient learning without major overfitting.

📊 Comparative Insights
Model Type	Architecture	Accuracy	Key Insight
FCNN	3 Dense Layers	97.6%	Learns global patterns but lacks spatial awareness
CNN	Conv + Pooling + Dense	99.0%	Captures localized image features efficiently


🧠 Learning Outcomes

Learned how CNNs outperform traditional fully connected networks for visual tasks.

Implemented key layers — Conv2D, Pooling, Flatten, and Dense — using Keras and TensorFlow.

Understood training control using EarlyStopping and evaluation through accuracy metrics.

Observed model reliability with validation loss curves and assert statements ensuring >98.5% accuracy.

🏁 Conclusion

The CNN model successfully classified image patterns with 99% accuracy, proving its robustness for real-world computer vision tasks such as urban scene analysis (e.g., Chicago Downtown).
The experiment demonstrated CNN’s efficiency in extracting hierarchical image features, setting the foundation for advanced applications like object detection, semantic segmentation, and visual recognition systems.

📁 Files in This Folder

Report 3.docx
assignment_3.py
Assignment 3.ipynb
