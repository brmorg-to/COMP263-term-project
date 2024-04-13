#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('unzip brain_tumors.zip')


# In[2]:


get_ipython().system('pip install tensorflow_addons')
get_ipython().system('pip install vit_keras')


# In[3]:


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer, LayerNormalization, MultiHeadAttention
from keras.models import Model

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow_addons as tfa
from vit_keras import vit


# In[4]:


# Set the seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# In[5]:


# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20


# In[6]:


# Load the data
data_dir = 'brain_tumors'
class_names = os.listdir(data_dir)
num_classes = len(class_names)


# In[7]:


# Prepare the dataset
data = []
labels = []
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(class_dir, img_name), target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        data.append(img_array)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)


# In[8]:


# Plot sample images from each class
plt.figure(figsize=(10, 10))
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    image_files = os.listdir(class_dir)
    sample_image = random.choice(image_files)
    sample_image_path = os.path.join(class_dir, sample_image)
    img = plt.imread(sample_image_path)
    plt.subplot(1, num_classes, i+1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')
plt.show()


# In[9]:


# Print the number of images in each class
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    num_images = len(os.listdir(class_dir))
    print(f"Number of images in {class_name}: {num_images}")


# In[10]:


# Split the data into training, validation, and testing sets while maintaining class proportions
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# In[11]:


# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# # model

# In[12]:


vit_model = vit.vit_b32(
        image_size = IMG_HEIGHT,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = num_classes)


# In[13]:


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# In[14]:


model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(num_classes, 'softmax')
    ],
    name = 'vision_transformer')


# In[15]:


# Compile the model
model.compile(optimizer=tfa.optimizers.AdamW(
                learning_rate=1e-5, weight_decay=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


# Train the model
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))


# In[17]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')


# In[18]:


# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# In[19]:


# Classification report
print(classification_report(y_test, y_pred_classes))


# In[20]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:')
print(conf_matrix)


# In[21]:


# Loss curve
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[22]:


# Select five random indices from the test set
random_indices = np.random.choice(len(x_test), 5, replace=False)


# In[23]:


# Get the actual and predicted classes for the selected indices
actual_classes = y_test[random_indices]
predicted_classes = y_pred_classes[random_indices]


# In[24]:


# Define class labels
class_labels = ["glioma_tumor", "meningioma_tumor", "normal", "pituitary_tumor"]


# In[25]:


# Plot the images with their actual and predicted classes
plt.figure(figsize=(15, 5))
for i, idx in enumerate(random_indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx])  # Assuming x_test contains the images
    plt.title(f"Actual: {class_labels[actual_classes[i]]}\nPredicted: {class_labels[predicted_classes[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[25]:




