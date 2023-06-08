import numpy as np
from skimage import measure
import tensorflow as tf
from tensorflow import keras


def apply_grey_patch(image, top_left_x, top_left_y, patch_size, grey_value = 1):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = grey_value

    return patched_image


def single_blob_occlusion(img, label, model):
    #Occlusion Map occluding one "blob" at a time
    #apply non-overlapping sensitivity map
    sensitivity_map = np.zeros((img.shape[0], img.shape[1]))
    regions, num_regions = measure.label(img, background = 0, return_num=True)

    img_ex_blob = 0
    lowest_confidence = 1.
    # Iterate the patch over the image
    for region in range(1,num_regions+1):
        patched_image = np.array(regions, copy = True)
        patched_image[patched_image == region] = 0
        patched_image[patched_image > 0] = 1
        predicted_classes = model.predict(np.array([patched_image]))[0]
        confidence = predicted_classes[label]

        #save occlusion image with highest difference in confidence
        if confidence <= lowest_confidence:
            img_ex_blob = patched_image
            lowest_confidence = confidence

        # Save confidence for this specific patched image in map
        sensitivity_map[
            np.where(regions == region)[0],
            np.where(regions == region)[1]
        ] = confidence

    sensitivity_map[sensitivity_map==0] = 0

    return sensitivity_map, img_ex_blob

#apply non-overlapping sensitivity map
def occlusion_map(img, label, patch_size, model):
    
    sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[0], patch_size):
        for top_left_y in range(0, img.shape[1], patch_size):
            patched_image = apply_grey_patch(img, top_left_x, top_left_y, patch_size)
            predicted_classes = model.predict(np.array([patched_image]))[0]
            confidence = predicted_classes[label]
            
            # Save confidence for this specific patched image in map
            sensitivity_map[
                top_left_y:top_left_y + patch_size,
                top_left_x:top_left_x + patch_size,
            ] = confidence

    return sensitivity_map

#variation on the sensitivity map with overlapping patches
def overlapping_occlusion_map(img, label, patch_size, step_size, model):
    
    sensitivity_map = np.zeros((img.shape[0], img.shape[1]))
    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[0] - patch_size + 1, step_size):
        for top_left_y in range(0, img.shape[1] - patch_size + 1, step_size):
            patched_image = apply_grey_patch(img, top_left_x, top_left_y, patch_size)
            predicted_classes = model.predict(np.array([patched_image]))[0]
            confidence = predicted_classes[label]
            
            # Save confidence for this specific patched image in map
            sensitivity_map[
                top_left_y:top_left_y + patch_size,
                top_left_x:top_left_x + patch_size,
            ] = (sensitivity_map[
                top_left_y:top_left_y + patch_size,
                top_left_x:top_left_x + patch_size,] + confidence)/2

    return sensitivity_map

def make_gradcam_heatmap(img, model, last_conv_layer_idx = -6, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    model.layers[-1].activation = None
    img_array = np.array([img])
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.layers[last_conv_layer_idx].output, model.output]
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = tf.keras.preprocessing.image.array_to_img(np.expand_dims(heatmap, axis = -1))
    heatmap = heatmap.resize((img.shape[0], img.shape[1]))
    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
    return heatmap.squeeze()