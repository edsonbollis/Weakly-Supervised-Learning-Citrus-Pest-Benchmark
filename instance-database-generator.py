import shutil
import sys
import os

import imageio
import numpy as np
import cv2
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K, Input, Model
from keras.preprocessing import image
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from tensorflow.python.framework import ops
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append(".")
from EfficientNet import _preprocess_input
import random
K.set_session
from shutil import copy

classes = 2

weights = './batch_normalized_expoent_index_whole/run_1/weights-improvement-01-0.68.hdf5'

layer_name="top_activation"

#Change your path here
prefix = '/home/edsonbollis/work/edsonbollis/'


folder = prefix + 'production/CPB_v1/'

y_csv = folder + 'pests_train_original.csv'
y_validation_csv = folder + 'pests_validation_original.csv'
y_test_csv = folder + 'pests_test_original.csv'

new_folder = prefix + 'production/CPB_v1_cuts/'


H, W = 1200, 1200  # Input shape, defined by the model (model.input_shape)
# Define model here ---------------------------------------------------

labels= ['Negative', 'Mite']


def load_data(directory):
    load = pd.read_csv(directory,delim_whitespace=False,header=1,delimiter=',',
                       names = ['images', labels[0],labels[1]])

    class_vet = [labels[i] for i in load['Mite']]

    load['classes'] = np.array(class_vet)
    load['classes_id'] = load['Mite'].to_numpy()

    count = 0
    for i in load['images']:
        if os.path.isfile(folder + i):
            count += 1
        print("Images found:", count, " Images remaining:", len(load['images']) - count)
    return load


def load_model_first():

    print("shape=",(W, H, 3))
    input_tensor = Input(shape=(W, H, 3))  # this assumes K.image_data_format() == 'channels_last'

    from EfficientNet import EfficientNetB32
    initial_model = EfficientNetB32(input_tensor=input_tensor, default_resolution=W, weights=None,
                                    include_top=False, input_shape=(W, H, 3),
                                    spatial_dropout=True, dropout=0.4)

    op = GlobalAveragePooling2D()(initial_model.output)

    output_tensor = Dense(classes, activation='softmax', use_bias=True, name='Logits')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = keras.regularizers.l1_l2(0.3)


    # model.compile(optimizer=keras.optimizers.adadelta(lr=0.1, rho=0.9, epsilon=1e-7, decay=0.0005),
    #               loss='categorical_crossentropy',
    #               metrics=[categorical_accuracy])

    model.load_weights(weights)


    return model


# ---------------------------------------------------------------------

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(W, H))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = _preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    # g = tf.get_default_graph()
    g = K.get_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = load_model_first()
    return new_model


def guided_backprop(input_model, images):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    return cam


def grad_cam_batch(input_model, images, classes):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output,
                        np.dstack([range(images.shape[0]), classes])[0].astype(int))
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], W, H))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (W, H), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams

# This Patch-SaliMap was modified to work with batch.
# The base of the algorithm is the same
# Here it creates the the Grad-CAM Image and the Patches
def Patch_SaliMap(model, guided_model, img_path, cls=-1):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    patch_size = 400
    patch_size = patch_size // 2

    initial = 0
    batch_size = 10
    cuts=5
    img_name = []
    img_mite_location_x = []
    img_mite_location_y = []


    for n in range((len(img_path) - initial) // batch_size):
        preprocessed_input = []
        images = []
        for i, img_ in enumerate(img_path[initial:initial+batch_size]):
            preprocessed_input.append(load_image(folder + img_))
            images.append(img_)
            for j in range(cuts):
                img_name.append(img_)
        preprocessed_input = np.concatenate(preprocessed_input,axis=0)

        # gradcam_batch = np.zeros((batch_size,H,W))
        # for i in range(preprocessed_input.shape[0]):
        #     gradcam_batch[i, :, :] = grad_cam(model, np.expand_dims(preprocessed_input[i,:,:],axis=0), -1*(cls[initial+batch_size] - 1))
        gradcam_batch = grad_cam_batch(model, preprocessed_input, cls[initial:initial+batch_size])
        # gb_batch = guided_backprop(guided_model, preprocessed_input)
        # guided_gradcam_batch = gb_batch * gradcam_batch[..., np.newaxis]
        print("Grad-cam batch len: ", gradcam_batch.shape[0])
        for i in range(gradcam_batch.shape[0]):
            img_ = images[i]
            gradcam = gradcam_batch[i,:,:]
            cls_actual = cls[initial + i]
            # gb = gb_batch[i]
            # guided_gradcam = guided_gradcam_batch

            jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
            print("#### jetcam.shape",jetcam.shape)
            ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
            jetcam[ind[0]-10:ind[0]+10,ind[1]-10:ind[1]+10] = 0
            jetcam = (np.float32(jetcam) + load_image(folder + img_, preprocess=False)) / 2
            cv2.imwrite(new_folder + 'gradcam_' + img_, np.uint8(jetcam))
            # cv2.imwrite('guided_backprop_' + img_, deprocess_image(gb[0]))
            # cv2.imwrite('guided_gradcam_' + img_, deprocess_image(guided_gradcam[0]))

            for j in range(cuts):
                ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
                image_ = image.img_to_array(load_image(folder + img_, preprocess=False))

                if cls_actual == 1:
                    ind = (int((ind[0] / gradcam.shape[0]) * image_.shape[0]),
                           int((ind[1] / gradcam.shape[1]) * image_.shape[1]))
                else:
                    ind = (random.randint(patch_size,W-patch_size),random.randint(patch_size,H-patch_size))

                img_mite_location_x.append(ind[0])
                img_mite_location_y.append(ind[1])

                vet_ = [ind[0], ind[1]]
                if ind[0] - patch_size < 0:
                    vet_[0] -= ind[0] - patch_size
                if ind[1] - patch_size < 0:
                    vet_[1] -= ind[1] - patch_size
                if ind[0] + patch_size >= image_.shape[0]:
                    vet_[0] += image_.shape[0] - ind[0] - patch_size + 1
                if ind[1] + patch_size >= image_.shape[0]:
                    vet_[1] += image_.shape[1] - ind[1] - patch_size + 1

                image_ = image_[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size, :]
                cv2.imwrite(new_folder + 'cutted_' + str(j) + "_" + img_, np.uint8(image_))
                gradcam[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size] = -1e7

        initial += batch_size
        print("Image" + str(initial) + " from " + str(len(img_path)))

    preprocessed_input = []
    images = []
    for i, img_ in enumerate(img_path[initial:len(img_path)]):
        preprocessed_input.append(load_image(folder + img_))
        images.append(img_)
        for j in range(cuts):
            img_name.append(img_)
    preprocessed_input = np.concatenate(preprocessed_input, axis=0)

    gradcam_batch = grad_cam_batch(model, preprocessed_input, cls[initial:len(img_path)])
    # gb_batch = guided_backprop(guided_model, preprocessed_input)
    # guided_gradcam_batch = gb_batch * gradcam_batch[..., np.newaxis]

    print("Grad-cam batch len: ", gradcam_batch.shape[0])
    for i in range(gradcam_batch.shape[0]):
        img_ = images[i]
        gradcam = gradcam_batch[i,:,:]
        cls_actual = cls[initial + i]

        for j in range(cuts):
            ind = np.unravel_index(np.argmax(gradcam, axis=None), gradcam.shape)
            image_ = image.img_to_array(load_image(folder + img_, preprocess=False))

            if cls_actual == 1:
                ind = (int((ind[0] / gradcam.shape[0]) * image_.shape[0]),
                       int((ind[1] / gradcam.shape[1]) * image_.shape[1]))
            else:
                ind = (random.randint(patch_size, W - patch_size), random.randint(patch_size, H - patch_size))

            img_mite_location_x.append(ind[0])
            img_mite_location_y.append(ind[1])

            vet_ = [ind[0], ind[1]]
            if ind[0] - patch_size < 0:
                vet_[0] -= ind[0] - patch_size
            if ind[1] - patch_size < 0:
                vet_[1] -= ind[1] - patch_size
            if ind[0] + patch_size >= image_.shape[0]:
                vet_[0] += image_.shape[0] - ind[0] - patch_size + 1
            if ind[1] + patch_size >= image_.shape[0]:
                vet_[1] += image_.shape[1] - ind[1] - patch_size + 1

            image_ = image_[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size, :]
            cv2.imwrite(new_folder + 'cutted_' + str(j) + "_" + img_, np.uint8(image_))
            gradcam[vet_[0] - patch_size: vet_[0] + patch_size, vet_[1] - patch_size: vet_[1] + patch_size] = -1e7

    initial = len(img_path)
    print("Image" + str(initial) + " from " + str(len(img_path)))

    df = pd.DataFrame()
    print("Len name array",len(img_name))
    df['images'] = np.array(img_name)
    print("Len x locations", len(img_mite_location_x))
    df['mite_location_x'] = np.array(img_mite_location_x)
    print("Len y locations", len(img_mite_location_y))
    df['mite_location_y'] = np.array(img_mite_location_y)
    df.to_csv(path_or_buf=new_folder +'pests_database_location_validation.csv')



if __name__ == '__main__':

    # choose y_csv to create the training set
    # y_validation_csv to create the validation set
    # y_test_csv to create the test set
    file = y_csv

    x_ = load_data(file).sample(frac=1, random_state=5)

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)

    copy(file, new_folder)

    print("Input preprocess", x_.shape)

    model = load_model_first()
    guided_model = build_guided_model()

    # If you need try just some exemples
    # for i in x_['images'][:64]:
    #     shutil.copy(folder  + i , "./")

    Patch_SaliMap(model, guided_model, img_path=x_['images'].to_list(), cls=x_['classes_id'].to_list())



