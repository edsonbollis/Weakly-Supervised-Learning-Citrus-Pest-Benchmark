from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

import keras
from keras import Input
import numpy as np
import sys
sys.path.append('../..')
sys.path.append('.')
from keras.layers import GlobalAveragePooling2D, Dense
from keras import regularizers
from keras.metrics import categorical_accuracy
from keras.models import Model
from image_data_generator import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os.path
import pandas as pd
from EfficientNet import _preprocess_input

from keras import backend as K
sys.path.append('../..')

width, height = None, None
size = None, None

log_file = open("log_batch_400.log", "a")

mean = np.array([0.36993951, 0.4671942,  0.34614164])
std = np.array([0.14322622, 0.1613108,  0.14663414])

layers = 3

multply_factor=1
dividing_factor=2
value = 2
classes = 2
cuts_number = 5


prefix = '/home/edsonbollis/work/edsonbollis/'


folder = prefix + 'production/CPB_v1_cuts/'

y_csv = folder + 'pests_train_original.csv'
y_validation_csv = folder + 'pests_validation_original.csv'
y_test_csv = folder + 'pests_test_original.csv'


labels= ['Negative', 'Mite']


def load_data(directory):
    load = pd.read_csv(directory,delim_whitespace=False,header=1,delimiter=',',
                       names = ['images', labels[1],labels[0]])

    class_vet = [labels[i] for i in load['Mite']]

    load['classes'] = np.array(class_vet)
    load['classes_id'] = load['Mite'].to_numpy()

    print

    count = 0
    for i in load['images']:
        if os.path.isfile(folder + i):
            count += 1
        print("Images found:", count, " Images remaining:", len(load['images']) - count)
    return load

def change_crop(data):

    df = pd.DataFrame()
    class_vet = []
    class_number_vet = []
    images = []
    for i, line in data.iterrows():
        for j in range(cuts_number):
            class_vet.append(line['classes'])
            class_number_vet.append(line['classes_id'])
            images.append('cutted_'+ str(j) + "_" + line['images'])


    df['classes'] = np.array(class_vet)
    df['classes_id'] = np.array(class_number_vet)
    df['images'] = np.array(images)

    count = 0
    for i in df['images']:
        if os.path.isfile(folder + i):
            count += 1
    print("Images found:",count, " Images remaining:", len(df['images']) - count)
    return df


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def load_model_first():

    print("shape=",(width, height, layers))
    input_tensor = Input(shape=(width, height, layers))  # this assumes K.image_data_format() == 'channels_last'

    from EfficientNet import EfficientNetB32
    initial_model = EfficientNetB32(input_tensor=input_tensor, default_resolution=width, weights=None,
                                    include_top=False, input_shape=(width, height, 3),
                                    spatial_dropout=True, dropout=0.4)

    op = GlobalAveragePooling2D()(initial_model.output)

    output_tensor = Dense(classes, activation='softmax', use_bias=True, name='Logits')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l1_l2(0.3)


    model.compile(optimizer=keras.optimizers.adadelta(lr=0.1, rho=0.9, epsilon=0.9, decay=0.0005),
                  loss='categorical_crossentropy',
                  metrics=[categorical_accuracy, f1])


    model.summary()

    return model


def train_model(x, x_val, x_test, model_name=None):
    # checkpoint

    model = load_model_first()

    model.load_weights(model_name)

    batch_size = 30
    print("\n*******Training")
    print("\n*******Training", file=log_file)
    accuracy_train = confusion_mat(model, batch_size, x, file='prediction_train')
    # accuracy_train = 0
    print("\n*******Validation")
    print("\n*******Validation", file=log_file)
    accuracy_validation = confusion_mat(model, batch_size, x_val, file='prediction_val')
    # accuracy_validation = 0
    print("\n*******Test")
    print("\n*******Test", file=log_file)
    accuracy_test = confusion_mat(model, batch_size, x_test, file='prediction_test')
    del model
    return accuracy_train, accuracy_validation, accuracy_test
    # return accuracy_test

def preprocessing_function(x):
    x = _preprocess_input(x)
    # x = ((x/255) - mean) / std
    # x = (x / 255)
    return x

def confusion_mat(model, batch_size, x, file = None):

    print("###################Shape and len", len(x), x.shape, (width, height))

    imageDatagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    print("shape=", (width, height, layers))


    _datagen = imageDatagen.flow_from_dataframe(x, directory=folder, x_col='images', y_col='classes',
                                                class_mode='categorical', target_size=(width, height),
                                                classes=labels,
                                                batch_size=batch_size, shuffle=False)

    y_val = np.array(np.array(x['classes_id']))

    y_val = y_val.flatten()

    Y_val = y_val[[cuts_number*i for i in range(y_val.shape[0]//cuts_number)]]
    # Y_val = y_val

    # Confution Matrix and Classification Report
    y_pred = model.predict_generator(_datagen,steps = len(x)/batch_size, verbose=1, workers=8, use_multiprocessing=False)

    new_pred = np.zeros((y_pred.shape[0]//cuts_number,y_pred.shape[1]))
    for i in range(y_pred.shape[0]//cuts_number):
        sum_ = cuts_number
        pred = cuts_number*y_pred[cuts_number * i]
        for j in range(1,cuts_number):
            sum_ += cuts_number - j
            pred += (cuts_number - j)*y_pred[cuts_number*i + j]
        pred_tot = pred/sum_
        new_pred[i,:] = pred_tot

    y_pred = new_pred

    arg_max = np.argmax(y_pred, axis=1)

    val_max = np.max(y_pred, axis=1)

    mask = (val_max > 0.) * 1

    Y_pred = arg_max * mask

    if file != None:
        np.save(file, Y_pred)

    print(Y_val, Y_pred)

    if Y_val.shape[0] < Y_pred.shape[0]:
        Y_pred = Y_pred[:Y_val.shape[0]]
    else:
        Y_val = Y_val[:Y_pred.shape[0]]

    vet = confusion_matrix(Y_val, Y_pred, labels=[i for i in range(0,classes)])

    val = sum([instance[i] for i,instance in enumerate(vet)])
    val /= len(x)

    accuracy = accuracy_score(Y_val, Y_pred, normalize=True, sample_weight=None)
    accuracy_no_norm = accuracy_score(Y_val, Y_pred, normalize=False, sample_weight=None)

    print('Confusion Matrix', np.array(Y_pred).shape, np.array(Y_val).shape)
    print('Confusion Matrix', np.array(Y_pred).shape, np.array(Y_val).shape, file=log_file)

    print("Accuracy", str(accuracy), 'No norm', str(accuracy_no_norm))
    print("Accuracy", str(accuracy), 'No norm', str(accuracy_no_norm), file=log_file)
    print('Classification Report')
    print('Classification Report', file=log_file)

    report = classification_report(Y_val, Y_pred, target_names=labels)
    print(report)
    print(report, file=log_file)
    log_file.flush()

    return str(accuracy).replace('.', ',')



def dfprocessing_func(data):

        model_name, x, x_val, x_test = data[0], data[1], data[2], data[3]

        print("\n*********Model name:", model_name,
              "\t**********\n*********", datetime.datetime.now(), "\t**********\n")
        print("\n*********Model name:", model_name,
              "\t**********\n*********", datetime.datetime.now(), "\t**********\n", file=log_file)
        weights_name.append(model_name)
        log_file.flush()

        train_val, validation_val, test_val = train_model(x, x_val, x_test, model_name=model_name)
        return [train_val, validation_val, test_val]

if __name__ == '__main__':


    model_names = []
    for root, dirs, files in os.walk('.', topdown=False):
        for name in files:
            if "weights-improvement" in name:
                print(name)
                model_names.append(name)
    model_names.sort(key=lambda x: os.path.getmtime(x))

    weights_name = []
    validation_values = []
    train_values = []
    test_values = []

    print("Call args: ", len(sys.argv))
    if len(sys.argv) > 1:
        string = sys.argv[1].split('_')[-1]
        fold = int(string.replace('/', ''))
    else:
        fold = 1
    print("###### Fold: ", fold)

    x = load_data(y_csv).sample(frac=1, random_state=fold)
    x_val = change_crop(x[80 * x.shape[0] // 100:])
    x_ = change_crop(x[:80 * x.shape[0] // 100])
    x_test = change_crop(load_data(y_validation_csv).sample(frac=1, random_state=5))

    # x_val = load_data(y_validation_csv).sample(frac=1, random_state=5)
    # x_test = load_data(y_test_csv).sample(frac=1, random_state=5)
    # x_test = load_data(y_test_2_csv).sample(frac=1, random_state=5)

    print(x, x_val, x_test)

    width, height = 400, 400
    size = width, height
    for idx, model_name in enumerate(model_names):
        if idx < len(model_names) - 1: continue

        # multprocess to release data
        # p = multiprocessing.Pool(1)
        # ret = p.map(dfprocessing_func, [[model_name, x, y, x_val, y_val, x_test, y_test]])
        ret = dfprocessing_func([model_name, x_, x_val, x_test])
        print(ret)
        train_val, validation_val, test_val =  ret[0], ret[1], ret[2]
        # p.terminate()
        # p.join()


        validation_values.append(validation_val)
        train_values.append(train_val)
        test_values.append(test_val)
        # del x, y, x_val, y_val, x_test, y_test

        print('\n\n\n')



    for i in weights_name:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in train_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in validation_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    for i in test_values:
        print(i, end='\t')
        print(i, end='\t', file=log_file)
    print()
    print(file=log_file)
    log_file.flush()
    log_file.close()
