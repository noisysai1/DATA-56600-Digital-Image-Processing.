# Week 2 – Edge Detection and Image Enhancement

### 🧭 Overview
This week explores **edge detection techniques**, a core component of image analysis used to identify object boundaries and significant transitions in intensity. 
Various methods—ranging from simple gradient-based filters to advanced multi-stage approaches—were implemented and compared for their performance on different images.

---

## 🧩 Key Concepts

### 1️⃣ Edge Detection Basics
- **Edge detection** identifies points in an image where intensity changes sharply.
- These changes usually represent object boundaries, textures, or feature outlines.
- Common approaches include **gradient filters (Sobel)** and **multi-stage algorithms (Canny)**.

---

## ⚙️ Implemented Techniques

### 🔹 1. Gradient-Based Edge Detection
A basic implementation that uses **horizontal and vertical gradient filters** to detect edges by computing pixel intensity differences.

**Algorithm:**
1. Apply horizontal and vertical convolution filters.  
2. Compute gradient magnitude:  
   \[
   G = \sqrt{(G_x^2 + G_y^2)}
   \]
3. Mark pixels exceeding a defined threshold as edges.

```python
import matplotlib.pyplot as plt
vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
img = plt.imread('bridge.png')
n, m, d = img.shape
edges_img = img.copy()
for row in range(3, n-2):
    for col in range(3, m-2):
        local_pixels = img[row-1:row+2, col-1:col+2, 0]
        v_score = (vertical_filter * local_pixels).sum() / 4
        h_score = (horizontal_filter * local_pixels).sum() / 4
        edge_score = (v_score**2 + h_score**2)**0.5
        edges_img[row, col] = [edge_score] * 3
edges_img = edges_img / edges_img.max()
plt.imshow(edges_img)
plt.title("Gradient-Based Edge Detection")
plt.show()


🔹 2. Sobel Edge Detection

The Sobel operator is a widely used method to highlight vertical and horizontal edges.

Steps:

Convert image to grayscale.

Apply Sobel X and Sobel Y filters.

Combine gradient outputs to form an edge map.


import cv2, numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

uploaded = files.upload()
img = cv2.imdecode(np.frombuffer(list(uploaded.values())[0], np.uint8), cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)
sobelxy = cv2.Sobel(img_blur, cv2.CV_64F, 1, 1, ksize=5)

cv2_imshow(sobelx)
cv2_imshow(sobely)
cv2_imshow(sobelxy)


🔹 3. Canny Edge Detection

The Canny algorithm provides accurate and noise-resistant edge detection through a multi-step process.

Stages:

Gaussian smoothing for noise reduction.

Gradient computation (magnitude and direction).

Non-maximum suppression to refine edges.

Hysteresis thresholding for strong and weak edges.


from skimage import io, feature
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
from io import BytesIO
import imageio

uploaded = files.upload()
filename = next(iter(uploaded))
image = Image.open(BytesIO(uploaded[filename])).convert('RGB')
image_array = rgb2gray(imageio.imread(BytesIO(uploaded[filename])))

edges = feature.canny(image_array)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(edges, cmap='viridis')
plt.title('Canny Edges')
plt.show()



🔹 4. Sobel + Gaussian Combined Filtering

This method first reduces image noise with a Gaussian filter and then applies Sobel filtering for edge enhancement.

Steps:

Apply Gaussian smoothing to suppress noise.

Use Sobel filters in both directions.

Compute gradient magnitude and mark edges above the threshold.

🔹 5. Gaussian Filter Detection

Applies a Gaussian filter directly to detect intensity transitions and highlight edges.


import skimage.io, skimage.filters
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

image = skimage.io.imread('bridge.png', as_gray=True)
def filter_function(sigma, threshold):
    masked = image.copy()
    masked[skimage.filters.gaussian(image, sigma=sigma) <= threshold] = 0
    plt.imshow(masked, cmap='gray')
    plt.axis('off')
    plt.show()

interact(filter_function, sigma=FloatSlider(min=0, max=7, step=0.1, value=2.5),
         threshold=FloatSlider(min=0, max=1, step=0.01, value=0.5))



📊 Observations and Results
Technique	Key Strength	Limitation
Gradient Filter	Simple and fast	Sensitive to noise
Sobel Operator	Detects both orientations	May blur small edges
Canny Detector	High accuracy and low noise sensitivity	Computationally expensive
Sobel + Gaussian	Balanced smoothness and sharpness	Fine-tuning required
Gaussian Filter	Smooth edge highlights	Over-smoothing may remove details


🧠 Learning Outcomes

Understood multiple edge detection algorithms and their use cases.

Implemented Sobel, Canny, and Gaussian filters using OpenCV and scikit-image.

Gained insight into balancing noise reduction vs. edge sharpness.

Built foundational understanding for segmentation and feature extraction tasks.



📁 Files in This Folder

Assignment2_Report.pdf

Assignment2.ipynb
