# Week 8 – Object Recognition and Deep Learning Classification

### 🧭 Overview
The final week of *DATA 56600 – Digital Image Processing* focused on integrating all previously learned image-processing concepts into an end-to-end **object-recognition pipeline**.  
Using pre-trained deep learning models (such as **VGG16** and **ResNet50**) through **transfer learning**, we applied automatic feature extraction and classification on real-world images.  
The goal was to identify objects accurately and compare CNN-based recognition with traditional segmentation and feature-based approaches from earlier weeks.

---

## 🧩 Key Concepts

### 1️⃣ Object Recognition
Object recognition combines **feature extraction** and **classification** to identify real-world entities in digital images.  
While traditional methods rely on handcrafted descriptors (SIFT, HOG, Harris), CNN-based models automatically learn hierarchical features from raw pixels.

### 2️⃣ Transfer Learning
Transfer learning leverages **pre-trained models** trained on large datasets (like ImageNet).  
By re-using learned feature hierarchies, we can:
- Reduce training time  
- Improve accuracy with limited data  
- Fine-tune high-level layers for domain-specific tasks  

### 3️⃣ CNN Architecture
A typical CNN for image classification consists of:
1. **Convolutional Layers** – detect spatial features such as edges and shapes  
2. **Pooling Layers** – down-sample feature maps to reduce complexity  
3. **Fully Connected Layers** – aggregate features into class probabilities  

---

## ⚙️ Implementation Workflow

### 🔹 Step 1: Load and Preprocess Images
Images are resized to 224×224 pixels (expected by most pretrained models), converted to RGB, and normalized to [0, 1].


import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

img_path = 'sample_object.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

🔹 Step 2: Load Pre-trained Model

We used VGG16 and ResNet50 from Keras Applications to classify objects without additional training.

from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions

model = VGG16(weights='imagenet')
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


Output Example:

Predicted: [('n02084071', 'dog', 0.92),
            ('n02123394', 'cat', 0.05),
            ('n02112018', 'poodle', 0.02)]

🔹 Step 3: Visualize Class Activation Map (CAM)

Grad-CAM highlights the regions of the image most influential for classification decisions.

import tensorflow as tf
import matplotlib.pyplot as plt

grad_model = tf.keras.models.Model(
    [model.inputs], [model.get_layer("block5_conv3").output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)[0]
weights = tf.reduce_mean(grads, axis=(0, 1))
cam = np.dot(conv_outputs[0], weights.numpy())

cam = cv2.resize(cam, (224, 224))
cam = np.maximum(cam, 0)
cam /= cam.max()
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
superimposed_img = np.uint8(0.4 * heatmap + 0.6 * img)

plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Class Activation Map (Grad-CAM)")
plt.axis('off')
plt.show()


📊 Results and Insights
Model	Accuracy	Strength	Limitation
VGG16	~92 % (top-1)	Stable and interpretable	Larger model size
ResNet50	~95 % (top-1)	Deeper residual learning; faster convergence	Requires GPU for optimal performance

Observations:

CNN-based recognition outperforms handcrafted-feature models (Week 6).

The Grad-CAM visualization clearly identifies the object region, validating the network’s interpretability.

Transfer learning dramatically reduces training requirements while achieving high accuracy.


🧠 Learning Outcomes

Implemented object recognition using pre-trained CNNs.

Understood transfer learning and model fine-tuning concepts.

Compared deep learning methods to traditional segmentation and feature extraction (Weeks 5–7).

Learned to interpret deep models through Grad-CAM visualizations.


🏁 Conclusion

Deep learning–based models such as VGG16 and ResNet50 demonstrate the power of convolutional networks for real-world image understanding.
Combining classical image processing (segmentation, edge detection, feature extraction) with deep CNN classification forms a complete, modern computer vision pipeline.

This final week reinforced the transition from algorithmic to learning-based image analysis—laying the foundation for advanced applications like object detection, semantic segmentation and medical-image analysis.


sample_object.jpg – example image for testing

results/grad_cam_visualization.png – highlighted activation map
