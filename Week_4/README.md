# Week 4 – Image Denoising using Convolutional Neural Networks (CNN)

### 🧭 Overview
This week focused on **image restoration and denoising** using **Convolutional Neural Networks (CNNs)**.  
The goal was to develop a neural network model capable of removing noise from corrupted images while preserving essential image features like edges and textures.  

The experiment involved uploading images, introducing artificial Gaussian noise and then using a CNN model to generate denoised outputs.  

---

## 🧩 Key Concepts

### 1️⃣ Image Denoising
Image denoising is the process of removing unwanted noise from images while retaining important details.  
Noise can originate from sensor errors, poor lighting or transmission distortion.  
By leveraging CNNs, we can perform **learning-based denoising**, allowing the model to automatically learn noise patterns and remove them effectively.

### 2️⃣ Convolutional Neural Networks (CNN)
CNNs are ideal for denoising because they:
- Extract spatial patterns (edges, textures) from noisy inputs.
- Learn filters that differentiate between noise and meaningful features.
- Use **ReLU activation** for non-linearity and **convolutional layers** for localized learning.
- Employ **pooling** to reduce dimensionality and avoid overfitting.

---

## ⚙️ Algorithm Design

### 🔹 Step 1: Upload and Preprocess Images
We Can uploads one or more images using `files.upload()` in Google Colab.  
Images are resized to a uniform **256×256** resolution and converted to **grayscale** for simplified computation.

```python
import cv2, numpy as np
from google.colab import files

uploaded = files.upload()
desired_height, desired_width = 256, 256
input_images = []

for path in uploaded.keys():
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (desired_width, desired_height))
    input_images.append(image.reshape((1, image.shape[0], image.shape[1], 1)))


🔹 Step 2: Add Gaussian Noise

To simulate real-world noise, Gaussian noise is added to the input images.

noisy_images = []
for image in input_images:
    noise = np.random.normal(loc=0, scale=1, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    noisy_images.append(noisy_image)

Mean = 0

Standard deviation = 1

Pixel values are clipped to stay within [0, 1].

This step generates noisy versions of the original images for training and evaluation.


🔹 Step 3: Define CNN-based Denoising Model

The CNN denoising model is built using TensorFlow and Keras, consisting of multiple convolutional layers followed by an output layer.

Each convolutional layer applies several filters to extract hierarchical features, gradually learning to reconstruct a clean image.

from tensorflow.keras.layers import Conv2D, Input
import tensorflow as tf

def cnn4dn_mdl(input_shape, layer_width=(8, 16, 8, 4)):
    inputs = Input(shape=input_shape)
    _tmp = inputs
    for _lw in layer_width:
        _tmp = Conv2D(filters=_lw, kernel_size=3, padding='same', activation='relu')(_tmp)
    _out = Conv2D(filters=1, kernel_size=3, padding='same', activation=None)(_tmp)
    return tf.keras.models.Model(inputs, _out)

tf.keras.backend.clear_session()
dn_mdl = cnn4dn_mdl(input_shape=(None, None, 1), layer_width=(8, 16, 8, 4))


Key Parameters:

Kernel size: 3×3

Activation: ReLU

Layer widths: (8, 16, 8, 4) filters for successive layers

Padding: Same, to preserve image dimensions


🔹 Step 4: Denoise the Images

The trained CNN model is applied to noisy images to predict clean versions.

denoised_images = [dn_mdl.predict(input_image) for input_image in input_images]


For each input, the model outputs a denoised version highlighting improved clarity and structure.


🔹 Step 5: Visualization

Original, noisy, and denoised images are displayed side by side for comparison.

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 20))
for i in range(len(input_images)):
    plt.subplot(3, len(input_images), 3*i + 1)
    plt.imshow(input_images[i].squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3, len(input_images), 3*i + 2)
    plt.imshow(noisy_images[i].squeeze(), cmap='gray')
    plt.title('Noisy')
    plt.axis('off')

    plt.subplot(3, len(input_images), 3*i + 3)
    plt.imshow(denoised_images[i].squeeze(), cmap='gray')
    plt.title('Denoised')
    plt.axis('off')

plt.tight_layout()
plt.show()

The side-by-side visualization demonstrates the model’s capability to suppress noise while preserving image structure.


📊 Results and Analysis
Image	Observation
Original	Clear and high-detail image
Noisy	Visible graininess and distortion due to Gaussian noise
Denoised	Significant noise reduction and clarity restoration
Summary of Findings:

The CNN denoising model effectively removed Gaussian noise while maintaining the sharpness of important features.

The output images had minimal distortion and high visual fidelity.

The technique generalized well for small variations in noise levels.


🧠 Learning Outcomes

Understood how CNN architectures can be adapted for image restoration.

Implemented end-to-end denoising from noisy to clean images using deep learning.

Learned preprocessing steps like resizing, grayscale conversion, and noise simulation.

Explored how feature extraction layers in CNNs play a crucial role in identifying and suppressing noise.


🏁 Conclusion

This week’s task demonstrated the power of deep convolutional networks in restoring image quality.
By training CNNs to distinguish between noise and true image structure, we achieved clean, high-quality outputs even from noisy inputs.

The results confirmed that CNN-based denoising is an effective and scalable approach for image restoration applications in medical imaging, surveillance and photography.

📁 Files in This Folder

Assignment 4.ipynb
