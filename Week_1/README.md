# Week 1 – Introduction to Digital Image Processing

### 🧭 Overview
This week introduces two fundamental concepts in digital image processing — **Sampling** and **Quantization**. 
These are the building blocks of digital image representation, determining how continuous real-world images are captured, stored, and displayed digitally.

---

## 🧩 Key Concepts

### 1️⃣ Sampling
Sampling is the process of converting a continuous image into a discrete one by selecting specific pixel locations. It determines the **spatial resolution** of an image.

- A **higher sampling rate** captures more pixel information but increases file size.
- A **lower sampling rate** reduces detail, causing blurring and pixelation.
- Sampling directly affects image clarity and storage efficiency.

### 2️⃣ Quantization
Quantization assigns discrete intensity levels to the sampled pixels. It determines the **color or grayscale resolution** of an image.

- Uses a limited number of bits per pixel to represent intensity levels.
- **Higher bit depth** → smoother gradients and better quality.
- **Lower bit depth** → loss of color accuracy and visible banding.

---

## ⚙️ Algorithm Design

### 🔹 Quantization Code
```python
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded.keys()))
image = io.imread(image_path)

print("Original image:")
plt.imshow(image)
plt.show()

ratio = 130  # Controls intensity compression
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            image[i][j][k] = int(image[i][j][k] / ratio) * ratio

print("Quantized image:")
plt.imshow(image)
plt.show()


🔹 Sampling Code

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded.keys()))
image = io.imread(image_path)

ratio = 20  # Defines the sampling interval
image1 = np.zeros((int(image.shape[0]/ratio),
                   int(image.shape[1]/ratio),
                   image.shape[2]), dtype='float32')

for i in range(image1.shape[0]):
    for j in range(image1.shape[1]):
        for k in range(image1.shape[2]):
            delta = image[i*ratio:(i+1)*ratio, j*ratio:(j+1)*ratio, k]
            image1[i, j, k] = np.mean(delta)

plt.imshow(image1.astype('uint8'))
plt.title("Sampled Image")
plt.show()

plt.imshow(image)
plt.title("Original Image")
plt.show()


📊 Observations and Results
🧮 Sampling Results

Noticeable loss in resolution and increased blur due to reduced pixel density.
Image appears smoother but less detailed.
Smaller file size, ideal for storage-efficient applications.
Trade-off: balancing data reduction with acceptable visual quality.

🎨 Quantization Results

Reduction in color range and subtle banding effects.
Visible distortion in gradients and fine textures.
File size decreases with fewer intensity levels.
Trade-off: smaller storage vs. loss of fidelity.

🧠 Learning Outcomes

Understood the distinction between spatial sampling and intensity quantization.
Implemented algorithms in Python using NumPy, Matplotlib, and scikit-image.
Visualized the impact of sampling and quantization on image quality.
Gained foundational knowledge for future topics like image enhancement, restoration, and segmentation.


📁 Files in This Folder

quantization_code.ipynb

sampling_code.ipynb

quantization_samples/

sampling_samples/

Assignment1_Report.pdf
