import os
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, precision_recall_curve, f1_score, jaccard_score

from DataGen import *
from Tlayers import *
import TUNET
import Cloudseg
import FCN
import Seg_Net

# check tensorflow version
print("tensorflow version:", tf.__version__)
# check available gpu
gpus = tf.config.list_physical_devices('GPU')
print("available gpus:", gpus)
# limit the gpu usage, prevent it from allocating all gpu memory for a simple model
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# check number of cpus available
print("available cpus:", os.cpu_count())

# Setting up parameters
image_size = 300  # 128 or 300
batch_size = 16
val_data_size = 600

# Dropout parameters for regularization1
input_dropout_rate = 0.8
dropout_rate = 0.5

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Training path having images and ground truths in different directories "images" and "GTmaps" 
#train_path = parent_dir + "/Datasets/SWINySEG/"
train_path = parent_dir + "/Datasets/TCDD/"

# Get image, GT ids. SWINySEG has .jpg images and .png GTs. To refer to a pair with same id, removing file extensions
lst = os.listdir(train_path + "images/")
train_ids = [x.split('.')[0] for x in lst]

# Divide ids into training and validation 
valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

# print(train_ids)

# Setting up the model
"""TUNET.image_size = image_size
TUNET.input_dropout_rate = input_dropout_rate
TUNET.dropout_rate = dropout_rate
model = TUNET.UNet()"""

## Second
model = Cloudseg.Cloudseg(image_size)

## Third
#model = FCN.FCN_model(image_size, dropout_rate=0.2)

## Fourth
#model = Seg_Net.SegNet(image_size)

# Learning rate for adam
learning_rate = "0.0005"

opt = keras.optimizers.Adam(learning_rate=float(learning_rate))
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
model.summary()

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

# Number of epochs
num_epochs = 130

history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=num_epochs)


epochs = range(1, num_epochs + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']


df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': train_loss,
    'Validation Loss': val_loss,
    'Train Accuracy': train_acc,
    'Validation Accuracy': val_acc
})

# swinyseg Dataset
# df.to_csv('Ttraining_history.csv', index=False)
#df.to_csv('CloudSeg_training_history.csv', index=False)
# df.to_csv('FCN_training_history.csv', index=False)
#df.to_csv('Seg_training_history.csv', index=False)

# TCDD Dataset
#df.to_csv('NewTtraining_history.csv', index=False)
df.to_csv('NewCloudSeg_training_history.csv', index=False)
#df.to_csv('NewFCN_training_history.csv', index=False)
#df.to_csv('NewSeg_training_history.csv', index=False)

print("Training history saved to 'training_history.csv'")


val_preds = []
val_labels = []

for i in range(valid_steps):
    x_val, y_val = valid_gen[i]
    preds = model.predict(x_val, verbose=0)
    val_preds.append(preds)
    val_labels.append(y_val)

# Convert lists to numpy arrays
val_preds = np.concatenate(val_preds, axis=0)
val_labels = np.concatenate(val_labels, axis=0)

# Binarize predictions and labels (assuming binary classification, threshold = 0.5)
val_preds = (val_preds > 0.5).astype(np.int32).flatten()
val_labels = (val_labels > 0.5).astype(np.int32).flatten()

val_recall = recall_score(val_labels, val_preds)
print("Recall Score on Validation Data: ", val_recall)

val_accuracy = accuracy_score(val_labels, val_preds)
print("Accuracy Score on Validation Data: ", val_accuracy)

val_f1 = f1_score(val_labels, val_preds)
print("F1 Score on Validation Data: ", val_f1)

val_iou = jaccard_score(val_labels, val_preds)
print("Mean IoU on Validation Data: ", val_iou)

val_error_rate = 1 - val_accuracy
print("Error Rate on Validation Data: ", val_error_rate)



train_preds = []
train_labels = []

for i in range(train_steps):
    x_train, y_train = train_gen[i]
    preds = model.predict(x_train, verbose=0)
    train_preds.append(preds)
    train_labels.append(y_train)

# Convert lists to numpy arrays
train_preds = np.concatenate(train_preds, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

# Binarize predictions and labels (assuming binary classification, threshold = 0.5)
train_preds = (train_preds > 0.5).astype(np.int32).flatten()
train_labels = (train_labels > 0.5).astype(np.int32).flatten()

train_recall = recall_score(train_labels, train_preds)
print("Recall Score on train Data: ", train_recall)

train_accuracy = accuracy_score(train_labels, train_preds)
print("Accuracy Score on train Data: ", train_accuracy)

train_f1 = f1_score(train_labels, train_preds)
print("F1 Score on train Data: ", train_f1)

train_iou = jaccard_score(train_labels, train_preds)
print("Mean IoU on train Data: ", train_iou)

train_error_rate = 1 - train_accuracy
print("Error Rate on train Data: ", train_error_rate)



# save weights swinyseg
# model.save_weights(current_dir + "/NewWeights/" + "TUNET.h5")
#model.save_weights(current_dir + "/NewWeights/" + "Cloudseg.h5")
# model.save_weights(current_dir + "/NewWeights/" + "FCN.h5")
#model.save_weights(current_dir + "/NewWeights/" + "Seg.h5")

# save weights TCDD
#model.save_weights(current_dir + "/NewWeights/" + "NewTUNET.h5")
model.save_weights(current_dir + "/NewWeights/" + "NewCloudseg.h5")
#model.save_weights(current_dir + "/NewWeights/" + "NewFCN.h5")
#model.save_weights(current_dir + "/NewWeights/" + "NewSeg.h5")

