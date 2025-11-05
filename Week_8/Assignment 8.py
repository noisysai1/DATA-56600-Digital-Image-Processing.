"""
Week 8 – Object Recognition using Pre-trained CNN
DATA 56600 – Digital Image Processing
Author: Sai Kumar Murarishetti
Lewis University
"""

# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --------------------------------------------------
# 2. Load and preprocess sample image
# --------------------------------------------------
img_path = 'sample_object.jpg'  # replace with your image filename
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# --------------------------------------------------
# 3. Load pre-trained VGG16 model
# --------------------------------------------------
print("Loading VGG16 pre-trained model...")
model = VGG16(weights='imagenet')
print(model.summary())

# --------------------------------------------------
# 4. Predict object class
# --------------------------------------------------
preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]
print("\nTop 3 Predictions:")
for i, (imagenetID, label, prob) in enumerate(decoded):
    print(f"{i + 1}. {label} ({prob * 100:.2f}%)")

# --------------------------------------------------
# 5. Visualize input image and Grad-CAM heatmap
# --------------------------------------------------
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

# Process Grad-CAM for visualization
cam = cv2.resize(cam, (224, 224))
cam = np.maximum(cam, 0)
cam = cam / cam.max()
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (224, 224))
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM Heatmap – {decoded[0][1]}")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n✅ Object Recognition completed successfully.")
