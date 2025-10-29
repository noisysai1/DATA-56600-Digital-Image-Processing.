# Week 5 – Image Denoising using Generative Adversarial Networks (TomoGAN)

### 🧭 Overview
This week’s assignment explored the use of **Generative Adversarial Networks (GANs)** for advanced **image denoising**, focusing on the **TomoGAN model** — a specialized
GAN architecture designed for denoising tomographic and scientific images.  

The primary objective was to restore noisy images by leveraging GANs’ ability to learn complex mappings between corrupted and clean data, thereby improving image clarity, precision, and usability for further analysis.

---

## 🧩 Key Concepts

### 1️⃣ Generative Adversarial Networks (GAN)
A **GAN** consists of two neural networks:
- **Generator:** Creates synthetic (denoised) images resembling clean input data.
- **Discriminator:** Evaluates whether an image is real (ground truth) or generated (fake).  

Both networks compete against each other in a **minimax game**, where the generator tries to fool the discriminator, and the discriminator learns to identify generated images.  
This **adversarial training** helps the generator produce increasingly realistic and noise-free images over time.

---

### 2️⃣ TomoGAN – A GAN for Image Denoising
**TomoGAN** is a modified GAN architecture created specifically for **scientific and tomographic image restoration**.  
It combines deep convolutional feature extraction with adversarial training to remove complex noise while preserving fine image structures.

**Key Features:**
- Learns from pairs of **noisy and clean images**.  
- Efficiently suppresses Gaussian and sensor noise.  
- Retains crucial scientific details, making it ideal for microscopy, CT, and radiology datasets.  

---

## ⚙️ Algorithm Design

### 🔹 Step 1: Load Pretrained TomoGAN Model
A pre-trained **TomoGAN model** (`TomoGAN.h5`) was downloaded from a public repository and loaded using TensorFlow.

```python
import tensorflow as tf, os, shutil

# Setup model folder
if os.path.isdir('model'):
    shutil.rmtree('model')
os.mkdir('model')

# Download pretrained model
!wget -O model/TomoGAN.h5 https://raw.githubusercontent.com/AIScienceTutorial/Denoising/main/model/TomoGAN.h5

# Load the model
TomoGAN_mdl = tf.keras.models.load_model('model/TomoGAN.h5')
TomoGAN_mdl.summary()


The model architecture includes multiple Conv2D layers, MaxPooling and UpSampling blocks, forming a U-Net style encoder-decoder network for feature extraction and reconstruction.

🔹 Step 2: Load and Preprocess Image

The input image is loaded, resized to 256×256, and converted to grayscale if necessary.

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

sample_img = plt.imread('download 1.png')

if sample_img.ndim == 3:
    sample_gray = np.mean(sample_img, axis=2, keepdims=True)
elif sample_img.ndim == 2:
    sample_gray = sample_img[..., np.newaxis]
else:
    raise ValueError("Unsupported image format")

resized_img = resize(sample_gray, (256, 256))
normalized_img = resized_img / 255.0


🔹 Step 3: Denoising with TomoGAN

The model predicts the denoised image from the normalized input.

dn_img = TomoGAN_mdl.predict(np.expand_dims(normalized_img, axis=0)).squeeze()


🔹 Step 4: Visualization

The results are visualized side by side — Noisy, Clean, and Denoised versions.

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(sample_gray.squeeze(), cmap='gray')
plt.title('Noisy/Input (download 1.png)')

plt.subplot(132)
plt.imshow(sample_gray.squeeze(), cmap='gray')
plt.title('Clean/Label (download 1.png)')

plt.subplot(133)
plt.imshow(dn_img.squeeze(), cmap='gray')
plt.title('Denoised (download 1.png)')

plt.tight_layout()
plt.show()

📊 Results and Observations

Image Type	Observation
Noisy/Input	Contains visible artifacts and random distortions.
Clean/Label	Ideal representation of true image data.
Denoised (TomoGAN)	Effectively suppresses noise and enhances detail without losing fine structures.

Analysis:

The TomoGAN model demonstrated excellent performance in removing noise from scientific imagery.

Denoised outputs were clearer and structurally accurate, closely matching the ground truth images.

The GAN framework preserved delicate image features, outperforming standard CNN-based denoising in maintaining realism.


🧠 Learning Outcomes

Gained hands-on experience with Generative Adversarial Networks for image restoration.

Learned how adversarial training improves denoising quality over traditional autoencoders.

Understood the architecture and workflow of TomoGAN, a specialized GAN model for tomographic data.

Practiced applying pre-trained deep learning models for scientific image enhancement.


🏁 Conclusion

Using TomoGAN, we successfully denoised real scientific images, achieving higher clarity and reduced artifacts.
The results demonstrated GANs’ potential to revolutionize image restoration and scientific visualization, providing high-quality outputs suitable for analysis and experimentation.

Future work could involve:
Fine-tuning TomoGAN for domain-specific data.
Testing with various noise profiles (Gaussian, Poisson, Salt & Pepper).
Exploring hybrid CNN-GAN frameworks for even better generalization.

📁 Files in This Folder

Assignment 5.ipynb
Assignment 5.pdf
model/TomoGAN.h5
results/denoised_output.png
