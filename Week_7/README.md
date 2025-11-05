# Week 7 – Image Segmentation (K-means, Mean Shift, Contours, Thresholding, Color Masking)

### 🧭 Overview
This week focuses on classical **image segmentation** pipelines that split an image into meaningful regions using **clustering (K-means, Mean Shift)**, **edge/contour extraction**, **thresholding (global/Otsu/adaptive)**, and **color masking in HSV**. 
We compare the approaches on natural images and visualize intermediate outputs.

---

## 🧩 Techniques Implemented
- **K-means classifying & segmentation** – groups pixels by color similarity in RGB space and assigns each pixel to the nearest cluster center; reshapes labels back to an image to visualize segments. 
- **Mean Shift (conceptual)** – cluster centers shift toward local density maxima; does not require K, useful when the number of regions is unknown.
- **Contour detection** – Canny edge detection (Gaussian smoothing → gradient magnitude/direction → non-max suppression → hysteresis) followed by `findContours()` to trace object boundaries.
- **Thresholding** – global and **Otsu** threshold selection to produce binary masks for foreground/background separation; adaptive thresholding noted for non-uniform illumination. 
- **Color masking (HSV)** – convert RGB→HSV, select hue/saturation/value ranges (e.g., blue), create mask via `inRange`, and apply it to isolate regions by color.

---

## ⚙️ Algorithm Design (high level)
1. **Load & preprocess**: read image(s), convert BGR↔RGB as needed.  
2. **K-means**: reshape image (H·W×3), run K-means, map labels to colors to view segments.  
3. **Contours**: grayscale → blur/threshold → Canny → `findContours` → draw/filled masks.  
4. **Thresholding**: compute Otsu threshold on grayscale; filter RGB with the binary mask.  
5. **Color mask (HSV)**: convert to HSV, select lower/upper bounds, `inRange`, bitwise-AND.  
6. **Compare** methods side-by-side on the same image(s).

---

## 🧑‍💻 Key Code 

### 1) K-means Segmentation
```python
import cv2, numpy as np

img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
pixels = np.float32(img.reshape(-1, 3))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K, attempts = 2, 10
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
segmented = centers[labels.flatten()].reshape(img.shape)  # color-coded segments
label_map = labels.reshape(img.shape[:2])                 # class map (grayscale)


(Mirrors the notebook’s K-means routine and plotting of both the colorized segments and label map.) 



2) Contour Detection & Region Masking
resized = cv2.resize(img, (256, 256))
gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2],
             key=cv2.contourArea)[-1]
mask = np.zeros(gray.shape, np.uint8)
mask = cv2.drawContours(mask, [cnt], -1, 255, -1)         # filled largest contour

segmented_region = cv2.bitwise_and(resized, resized, mask=mask)




3) Otsu Thresholding + RGB Filtering
from skimage.filters import threshold_otsu
import numpy as np

g = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
t = threshold_otsu(g)
mask_otsu = g < t

def apply_mask(rgb, m):
    return np.dstack([rgb[...,0]*m, rgb[...,1]*m, rgb[...,2]*m])

filtered = apply_mask(resized, mask_otsu)




4) HSV Color Masking (example: blue)
hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
lower_blue = (90, 70, 50)
upper_blue = (128, 255, 255)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
blue_regions = cv2.bitwise_and(resized, resized, mask=mask_blue)




📊 Results & Comparison

Method	Strengths	Limitations / Notes
K-means (K=2)	Fast, unsupervised color grouping	Must choose K; ignores spatial context
Mean Shift (conceptual)	Finds natural modes; no preset K	Heavier compute on large images
Contour detection	Precise borders for prominent objects	Needs clean edges; sensitive to thresholds
Otsu thresholding	Automatic global threshold	Struggles with non-uniform illumination
HSV color masking	Intuitive color isolation via hue/sat/value ranges	Requires tuning bounds per image/lighting

The figure below (from this repo) shows the pipeline outputs on a landscape image:

/Week7_ImageSegmentation/outputs/download.png

(Grid: Original • K-means Segmentation • K-means Classifying • Contour Detection • Segmented Regions • Thresholding • Color Masking.) 



🧠 Learning Outcomes

Implemented and compared five segmentation strategies on real images.

Learned how clustering, edges/contours, binary masks, and HSV ranges each contribute to robust segmentation under different conditions. 



📁 Files in This Folder

Assignment 7 Report.pdf – methods, steps and analysis. 

Assignment 7.pdf / notebook_code.py – executable code blocks for K-means, contours, Otsu, HSV. 

Assignment 7

inputs/ – sample images (Img1.jpeg, Img2.jpeg, …)

outputs/download.png – montage of all intermediate results (shown above).
