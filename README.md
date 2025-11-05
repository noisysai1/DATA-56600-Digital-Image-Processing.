# DATA-56600-Digital-Image-Processing.
Explores image enhancement, restoration, segmentation, feature extraction and object recognition using Python and OpenCV.

This repository documents my complete 8-week journey through **DATA 56600 – Digital Image Processing**.  
Each week explores a major theme in modern image analysis—from foundational sampling and quantization to advanced neural-network-based object recognition.  
All implementations were completed in Python using OpenCV, scikit-image, TensorFlow, and Keras.

---

## 🗂️ Weekly Modules Summary

| Week | Topic | Key Focus Areas |
|------|--------|----------------|
| **Week 1** | **Introduction to Digital Image Processing** | Sampling and quantization • Image representation • Pixel intensity operations • Basic filtering |
| **Week 2** | **Edge Detection Techniques** | Gradient methods (Sobel, Prewitt, Laplacian) • Canny edge detection • Gaussian smoothing and noise reduction |
| **Week 3** | **Image Classification with CNNs** | MNIST dataset • Building and training CNNs • Activation functions • Accuracy evaluation and visualization |
| **Week 4** | **Image Restoration and Denoising using CNNs** | Image noise modeling • Autoencoders • Loss functions (MSE, MAE) • Denoised image comparison |
| **Week 5** | **Generative Adversarial Networks (GANs) for Scientific Image Denoising** | TomoGAN architecture • Generator–Discriminator training loop • Noise suppression metrics |
| **Week 6** | **Feature Extraction and Classification** | Harris Corner Detection • SIFT • HOG • Texture-based feature representation • Object matching |
| **Week 7** | **Image Segmentation Techniques** | K-means • Mean Shift • Contours • Thresholding (Otsu/Adaptive) • HSV color masking • Object isolation |
| **Week 8** | **Object Recognition using Deep Learning (Transfer Learning)** | VGG16 and ResNet50 • Feature reuse • Grad-CAM visualization • Class prediction and interpretability |

---

## 🧠 What I Learned During This Course

This course provided me with both **theoretical understanding and practical experience** in digital image processing and computer vision.  
Throughout these 8 weeks, I learned how to analyze, process, and interpret visual data using both classical algorithms and deep learning frameworks.


### 🔹 Technical Skills Gained
- **Image Fundamentals:** Learned how images are represented as pixel matrices, and how sampling and quantization affect quality and size.  
- **Filtering & Enhancement:** Applied spatial and frequency domain filters for smoothing, sharpening, and contrast improvement.  
- **Edge Detection & Segmentation:** Used gradient-based and clustering methods to locate and isolate objects.  
- **Feature Engineering:** Extracted and matched features using algorithms like SIFT, HOG, and Harris corner detection.  
- **Machine Learning for Vision:** Built and trained **Convolutional Neural Networks (CNNs)** for image classification.  
- **Image Restoration & Denoising:** Implemented CNN and **TomoGAN** models for removing Gaussian noise and restoring clarity.  
- **Transfer Learning & Object Recognition:** Fine-tuned pre-trained networks (VGG16, ResNet50) and visualized **Grad-CAM** to interpret model decisions.  
- **Visualization & Analysis:** Developed comparative visualizations for different filters, segmentation results, and model accuracies using **Matplotlib and OpenCV**.  
- **Practical Implementation:** Strengthened Python coding proficiency and GPU-based model deployment in Google Colab.  


### 🔹 Conceptual Understanding
- Grasped the **mathematical foundation** of convolution, correlation, and transformation in image processing.  
- Understood **how neural networks “see” images** through hierarchical feature extraction.  
- Learned **evaluation metrics** for image-based tasks (accuracy, PSNR, MSE, SSIM).  
- Developed a **structured workflow** for designing and evaluating end-to-end vision systems.

---

## 🧩 Tools & Libraries Used
| Category | Libraries |
|-----------|------------|
| **Image Processing** | `OpenCV`, `scikit-image`, `NumPy`, `Matplotlib`, `Pillow` |
| **Deep Learning** | `TensorFlow`, `Keras`, `PyTorch (optional for GAN experiments)` |
| **Statistical Analysis** | `scikit-learn`, `pandas`, `seaborn` |
| **Environment** | Google Colab, Jupyter Notebook, VS Code |

---

## 🧪 Sample Project Outputs

| Week | Visualization |
|------|----------------|
| Week 1 | Sampling & Quantization comparison |
| Week 2 | Sobel vs Canny edge detection |
| Week 3 | CNN accuracy curves & confusion matrix |
| Week 4 | Original vs Noisy vs Denoised images |
| Week 5 | TomoGAN-generated denoised outputs |
| Week 6 | Feature keypoints via SIFT & HOG |
| Week 7 | K-means segmentation & color masking |
| Week 8 | Grad-CAM visualization on VGG16 |

---

## 🧠 Conceptual Progression

1. **Weeks 1–2:** Built strong foundations in image representation and enhancement.  
2. **Weeks 3–4:** Transitioned into machine learning and neural network architectures.  
3. **Week 5:** Introduced to generative models and adversarial training for denoising.  
4. **Weeks 6–7:** Focused on region-based and clustering segmentation methods.  
5. **Week 8:** Integrated all knowledge into a complete object-recognition pipeline using transfer learning.

---

## 📁 Repository Structure

Digital-Image-Processing/
│
├── Week1_Introduction/
├── Week2_EdgeDetection/
├── Week3_CNN_Classification/
├── Week4_CNN_Denoising/
├── Week5_TomoGAN_Denoising/
├── Week6_FeatureExtraction/
├── Week7_ImageSegmentation/
├── Week8_ObjectRecognition/
│ ├── week8_object_recognition.py
│ ├── sample_object.jpg
│ └── results/
│
└── README.md ←

---

## 🏁 Overall Learning Outcomes
- Developed the ability to **implement full image-processing pipelines** from raw data to classification.  
- Strengthened analytical skills for **interpreting pixel-level transformations and model behaviors**.  
- Learned to compare **traditional and AI-driven methods** for efficiency, accuracy and interpretability.  
- Acquired proficiency in **deep learning frameworks** for real-world computer vision applications.  
- Built a professional-grade **portfolio repository** that showcases technical and conceptual mastery of digital image processing.

---
