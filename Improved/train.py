import os
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, precision_recall_curve, f1_score, jaccard_score

from DataGen import *
from layers import *
import UNET

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
image_size = 128
batch_size = 6
val_data_size = 600

# Dropout parameters for regularization
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

#print(train_ids)

# Setting up the model
UNET.image_size = image_size
UNET.input_dropout_rate = input_dropout_rate
UNET.dropout_rate = dropout_rate
model = UNET.UNet()

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
#df.to_csv('training_history.csv', index=False)
#df.to_csv('OnlyD_Block_history.csv', index=False)
#df.to_csv('OnlyDs_Path_history.csv', index=False)
#df.to_csv('OnlyCSAM_history.csv', index=False)

# TCDD Dataset
df.to_csv('Newtraining_history.csv', index=False)
#df.to_csv('NewOnlyD_Block_history.csv', index=False)
#df.to_csv('NewOnlyDs_Path_history.csv', index=False)
#df.to_csv('NewOnlyCSAM_history.csv', index=False)
print("Training history saved to 'training_history.csv'")


val_preds = []
val_labels = []

for i in range(valid_steps):
    x_val, y_val = valid_gen[i]
    preds = model.predict(x_val, verbose=0)
    val_preds.append(preds)
    val_labels.append(y_val)

val_preds = np.concatenate(val_preds, axis=0)
val_labels = np.concatenate(val_labels, axis=0)


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

train_preds = np.concatenate(train_preds, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

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
#model.save_weights(current_dir + "/NewWeights/" + "MyModel.h5")
#model.save_weights(current_dir + "/NewWeights/" + "OnlyD_Block.h5")
#model.save_weights(current_dir + "/NewWeights/" + "OnlyDs_Path.h5")
#model.save_weights(current_dir + "/NewWeights/" + "OnlyCSAM.h5")

# save weights TCDD
model.save_weights(current_dir + "/NewWeights/" + "NewMyModel.h5")
#model.save_weights(current_dir + "/NewWeights/" + "NewOnlyD_Block.h5")
#model.save_weights(current_dir + "/NewWeights/" + "NewOnlyDs_Path.h5")
#model.save_weights(current_dir + "/NewWeights/" + "NewOnlyCSAM.h5")
