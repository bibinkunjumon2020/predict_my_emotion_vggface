
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras_vggface.vggface import VGGFace
# from keras.optimizers import SGD
from keras.optimizers.legacy import SGD
from keras.callbacks import ModelCheckpoint
import label_and_dir
from data_generator import data_generator
import data_downloading
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import mlflow
mlflow.tensorflow.autolog()

# Set the experiment name
mlflow.set_experiment("my-last-experiment")

#downloading data from Kaggle

# data_downloading.data_download()

#importing data
train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir.label_and_dir()
train_generator, valid_generator, test_generator = data_generator(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)

#custom parameters
nb_class = 7
hidden_dim = 1024

#Creating a VGGFace model instance.
vgg_model = VGGFace(include_top=False, input_shape=(96, 96, 3))

#Use the architecture of the VGGFace and append a fully connected layer with 1024 neurons before the final classification using softmax.
last_layer = vgg_model.get_layer('pool5').output
x = layers.Flatten()(last_layer)
x = layers.Dense(hidden_dim, activation='relu', name='fc7')(x)
x = layers.Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, x)

print("Model Summary\n")
#Printing the model summary
custom_vgg_model.summary()

# Training the model with fer2013 data.
custom_vgg_model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.0001),
              metrics=['accuracy'])

print("vgg_model compiled")
#Creating a callback that saves a model when the validation loss decreases from the previous epoch.
filepath="../trained_models2/weights-improvement-{epoch:02d}-{val_loss:.2f}"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)
callbacks = [checkpoint]

# my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2),
#     tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}'),
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
# ]

#training the model

print("Starting the model.fit: Epoch takes time ")

history = custom_vgg_model.fit(
    train_generator,
    validation_data=valid_generator,validation_split=0.1,
    # steps_per_epoch=897,
    epochs=30,
    # validation_steps=897,
    verbose=1,
    callbacks=callbacks,
)
custom_vgg_model.save("final_model/my_model2")

newmodel = tf.keras.models.load_model("final_model/my_model2")
print("Trained Model Summary")
newmodel.summary()
#prediction
predictions = newmodel.predict(test_generator)

predicted_labels = np.argmax(predictions, axis=1)

print("Predictions\n",predicted_labels,"\nLength = ",len(predicted_labels))

true_labels = test_label['emotion'].astype(int).values

print("True Labels\n",true_labels,"\nLength = ",len(true_labels))


f1 = f1_score(true_labels, predicted_labels, average='macro')
print("\n****F1 Score****\n")
print(f1)
# mlflow.log_metric("f1score",f1)

cm = confusion_matrix(true_labels, predicted_labels)
print("\n****Confusion_Matrix****\n")
print(cm)

print("-"*20)
autolog_run = mlflow.last_active_run()


####

newmodel = tf.keras.models.load_model("../datasets/trained_models/trained_vggface.h5",compile=False)

import cv2
image_path = "dq.jpeg"
img = cv2.imread(image_path)
cropped_img = cv2.resize(img, (96,96)) 
cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
cropped_img_float = cropped_img_expanded.astype(float)
prediction = newmodel.predict(cropped_img_float)
predicted_label = int(np.argmax(prediction))
print("*****",predicted_label,"******") # it is the emotion index in dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
print("Predicted Label:", predicted_label,emotion_dict.get(predicted_label))
