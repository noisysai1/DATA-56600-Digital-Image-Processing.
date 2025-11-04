# Week 6 – Feature Extraction and Classification

### 🧭 Overview
This week focused on **Feature Extraction** techniques, a critical step in image analysis and computer vision tasks. 
The goal was to extract meaningful information from images, such as edges, corners, and textures, which can be used for tasks like **image classification**, **object recognition**, and **image matching**. 
We explored methods like **Harris Corner Detection**, **SIFT** (Scale-Invariant Feature Transform), and **HOG** (Histogram of Oriented Gradients) to extract relevant features from images.

---

## 🧩 Key Concepts

### 1️⃣ Feature Extraction
Feature extraction involves transforming an image into a set of descriptors that represent key elements of the image, such as:
- **Edges:** Boundaries between regions with different intensity.
- **Corners:** Points where two edges meet, useful for object recognition.
- **Textures:** Patterns of pixel intensity that describe surface properties.

### 2️⃣ Harris Corner Detection
The **Harris Corner Detection** algorithm identifies **corner points** where the intensity gradient changes significantly in multiple directions. 
Corners are valuable for image matching because they are stable and distinctive.

### 3️⃣ Scale-Invariant Feature Transform (SIFT)
SIFT detects and describes **local features** in images that are invariant to scale, rotation, and partial illumination changes. I
t is widely used for **image matching** and **object recognition** in challenging environments.

### 4️⃣ Histogram of Oriented Gradients (HOG)
HOG is used to extract **gradient orientation histograms** in local cells of an image. It is particularly effective in object detection tasks, such as detecting people in images.

---

## ⚙️ Algorithm Design

### 🔹 Step 1: Harris Corner Detection
Harris Corner Detection is a feature extraction technique that detects corners in images. 
It computes the **gradient of the image** and uses these gradients to find where significant changes in the image intensity occur.

**Code Snippet:**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Apply Harris corner detection
corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)

# Result is dilated for marking the corners
corners = cv2.dilate(corners, None)

# Display the result
image[corners > 0.01 * corners.max()] = [255, 0, 0]
plt.imshow(image)
plt.title('Harris Corner Detection')
plt.show()

🔹 Step 2: SIFT Feature Detection

SIFT identifies distinctive features at various scales. The algorithm detects keypoints and describes them with robust feature descriptors.

Code Snippet:

import cv2
from matplotlib import pyplot as plt

# Load image and convert to grayscale
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Show the image with keypoints
plt.imshow(image_with_keypoints)
plt.title('SIFT Feature Detection')
plt.show()

🔹 Step 3: HOG Feature Extraction

The HOG descriptor counts occurrences of gradient orientation in localized portions of an image.

Code Snippet:

import cv2
import numpy as np
from skimage.feature import hog
from matplotlib import pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Compute HOG features and visualization
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Display the HOG image
plt.imshow(hog_image, cmap=plt.cm.gray)
plt.title('HOG Feature Extraction')
plt.show()

📊 Results and Observations

Feature Extraction Method	Key Benefit	Limitation
Harris Corner Detection	Effective for detecting corner points	Sensitive to noise and not invariant to scale/rotation
SIFT	Robust to scale, rotation, and partial illumination changes	Computationally expensive
HOG	Effective for object detection, especially in pedestrian detection	Sensitive to small image variations
Observations:

Harris Corner Detection performed well in identifying distinct points but was affected by noise.
SIFT detected stable and distinctive features even with scale and rotation changes.
HOG was very effective for recognizing shapes and detecting pedestrians in the images.


🧠 Learning Outcomes

Implemented Harris Corner Detection to identify strong corner features in images.

Used SIFT to detect and describe keypoints that are invariant to transformations.

Extracted HOG features to use for object detection tasks, such as pedestrian detection.

Gained a deeper understanding of how feature extraction can improve image classification and matching tasks.


🏁 Conclusion

Feature extraction is an essential step in many computer vision tasks, including object recognition, image matching, and classification. Techniques like Harris Corner Detection, SIFT, and HOG are widely used due to their robustness in capturing distinctive image features.

This week reinforced the importance of selecting the right feature extraction method for different image processing tasks. As we continue to explore more advanced topics, understanding these foundational techniques will prove valuable for tackling more complex tasks such as image segmentation and object tracking.
