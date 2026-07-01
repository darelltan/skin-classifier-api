
import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['basal_cell', 'keratosis', 'melanoma', 'nevus']

# Update these names if your model was rebuilt — check model.layers
GAP_LAYER      = 'global_average_pooling2d_2'
BN_LAYER       = 'batch_normalization_2'
DROPOUT_LAYER  = 'dropout_2'
DENSE_LAYER    = 'dense_2'

def get_prediction(img_batch, model):
    predictions = model(tf.cast(img_batch, tf.float32), training=False)
    pred_index  = int(np.argmax(predictions[0]))
    return {
        'class':      CLASS_NAMES[pred_index],
        'confidence': round(float(predictions[0][pred_index]) * 100, 1),
        'all_scores': {
            CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 1)
            for i in range(len(CLASS_NAMES))
        }
    }

def make_gradcam_heatmap(img_batch, model):
    efficientnet = model.get_layer('efficientnetb0')
    img_tensor   = tf.cast(img_batch, tf.float32)

    predictions = model(img_tensor, training=False)
    pred_index  = int(np.argmax(predictions[0]))

    feat_model = tf.keras.Model(
        inputs  = efficientnet.input,
        outputs = efficientnet.get_layer('top_conv').output
    )
    conv_out = feat_model(img_tensor, training=False)

    with tf.GradientTape() as tape:
        tape.watch(conv_out)
        x = conv_out
        after_conv = False
        for layer in efficientnet.layers:
            if after_conv:
                x = layer(x)
            if layer.name == 'top_conv':
                after_conv = True

        x = model.get_layer(GAP_LAYER)(x)
        x = model.get_layer(BN_LAYER)(x)
        x = model.get_layer(DROPOUT_LAYER)(x, training=False)
        x = model.get_layer(DENSE_LAYER)(x)
        class_output = x[:, pred_index]

    grads        = tape.gradient(class_output, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(tf.nn.relu(heatmap)).numpy()
    heatmap      = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def overlay_heatmap(original_arr, heatmap):
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_arr.astype(np.uint8), 0.6, h, 0.4, 0)
