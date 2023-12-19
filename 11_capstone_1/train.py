# import Deep learning Libraries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os

# import data handling tools
import numpy as np
import pandas as pd

##################################################################################

data_dir = './data/Training'
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)

        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
train_data = pd.concat([Fseries, Lseries], axis=1)

##################################################################################

data_dir = './data/Testing'
filepaths = []
labels = []

folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)

        filepaths.append(fpath)
        labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
test_data = pd.concat([Fseries, Lseries], axis=1)

##################################################################################

df = pd.concat([train_data, test_data])

strat = df['labels']
train_df, test_df = train_test_split(
    df,  train_size=0.8, random_state=42, stratify=strat)

##################################################################################

batch_size = 8
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator()
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=batch_size)

test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                      color_mode='rgb', shuffle=False, batch_size=batch_size)

##################################################################################

# to define number of classes in dense layer
class_count = len(list(train_gen.class_indices.keys()))


def make_model(learning_rate=0.001, size_inner=256, droprate=0.5):

    base_model = keras.applications.efficientnet_v2.EfficientNetV2B3(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling='max')

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=img_shape)
    base = base_model(inputs, training=False)
    batch = BatchNormalization(axis=-1,
                               momentum=0.99,
                               epsilon=0.001)(base)
    inner = Dense(size_inner,
                  kernel_regularizer=regularizers.l2(l=0.016),
                  activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006),
                  activation='relu')(batch)
    drop = Dropout(droprate)(inner)
    outputs = Dense(class_count, activation='softmax')(drop)
    model = keras.Model(inputs, outputs)

    #########################################

    optimizer = Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model

##################################################################################

checkpoint = ModelCheckpoint(
    'EfficientNetV2B3_v2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

learning_rate = 0.001
size = 128
droprate = 0.5

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(x=train_gen,
                    epochs=30,
                    validation_data=test_gen,
                    callbacks=[checkpoint, early_stopping])

##################################################################################

train_score = model.evaluate(train_gen, verbose=1)
test_score = model.evaluate(test_gen, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])


preds = model.predict_generator(test_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

print(classification_report(test_gen.classes, y_pred, target_names=classes))

##################################################################################