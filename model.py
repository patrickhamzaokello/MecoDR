import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

AUTOTUNE = tf.data.experimental.AUTOTUNE

trained_checkpoint = '../input/dr-model-checks/checkpoint'
sample_image_path = '../input/sample-image/cd563556cb57.png'

classes = {0: "No DR",
           1: "Mild",
           2: "Moderate",
           3: "Severe",
           4: "Proliferative"}

from PIL import Image

image_path = Image.open(sample_image_path)
plt.imshow(image_path)
plt.axis("off")

num_classes = 5
input_shape = (512, 512, 3)
learning_rate = 1e-4  # 0.001
weight_decay = 0.0001
batch_size = 16  # 256
num_epochs = 2
# We'll resize input images to this size
image_size = 256
# Size of the patches to be extract from the input images
patch_size = 7
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
# Size of the dense layers of the final classifier
mlp_head_units = [56, 28]  # [1024, 512]

# load

print("[INFO] loading model...")

vit_classifier = load_model("../input/trained-model")
vit_classifier.load_weights(trained_checkpoint)

# model summary
vit_classifier.summary()


# GRAD CAM DEF
def get_img_array(img):
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


classes.values()


def display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4, preds=[0, 0, 0, 0, 0]):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM


#     plot.imshow(superimposed_img)
#     plot.set(title =
#         " No DR: \
#         {:.3f}\nMild: \
#         {:.3f}\nModerate: \
#         {:.3f}\nSevere: \
#         {:.3f}\nProliferative: \
#         {:.3f}".format(preds[0], \
#                     preds[1], \
#                     preds[2], \
#                     preds[3],
#                     preds[4])
#     )
#     plot.axis('off')

#  test single image from file
# Import required libraries
import tensorflow as tf
from PIL import Image

# Read a PIL image
img = Image.open(sample_image_path)
# print(img)
# Convert the PIL image to Tensor
img_to_tensor = tf.convert_to_tensor(img)
# print the converted Torch tensor
print(img_to_tensor)
# print("dtype of tensor:",img_to_tensor.dtype)


# pridict image
img_array = get_img_array(img_to_tensor)
# Remove last layer's softmax
vit_classifier.layers[-1].activation = None

# check the model to ensure the layer name matches
last_conv_layer_name = 'layer_normalization_4'

# Print what the top predicted class is
preds = vit_classifier.predict(img_array)
heatmap = gradcam_heatmap(img_array, vit_classifier, last_conv_layer_name)

heatmap = np.reshape(heatmap, (36, 36))
display_gradcam(img_to_tensor, heatmap, preds=preds[0])

print(preds)
