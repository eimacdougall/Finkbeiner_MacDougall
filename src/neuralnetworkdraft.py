#all da imports

import tensorflow as tf
from keras import regularizers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import data
import evaluate
import features
import shap

def plot_image(i, predictions_array, true_label, img):
  """plot image and predictions"""
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(labels[predicted_label],
                                100*np.max(predictions_array),
                                labels[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Load data
train_images, train_labels = data.load_mnist('data\\fashion', kind='train')
test_images, test_labels = data.load_mnist('data\\fashion', kind='t10k')

#fix data shape (ensure low values)
train_images = train_images / 255.0
test_images = test_images / 255.0
# train_reg_images = train_reg_images / 255.0
# test_reg_images = test_reg_images / 255.0

train_reg_images, scaler = features.regression_preprocess(train_images)
test_reg_images = scaler.transform(test_images)

print(f"test shape: {test_images.shape}")

#ensure shape valid
train_images = train_images.reshape((-1, 28, 28))
test_images = test_images.reshape((-1, 28, 28))
train_reg_images = train_reg_images.reshape((-1, 28, 28, 1))
test_reg_images  = test_reg_images.reshape((-1, 28, 28, 1))

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]
train_reg_images = train_images[..., np.newaxis]
test_reg_images = test_images[..., np.newaxis]

print(f"test shape after reshape: {test_images.shape}")
print(f"train shape after reshape: {train_images.shape}")
print(f"reg train shape after reshape: {train_reg_images.shape}")
print(f"reg test shape after reshape: {test_reg_images.shape}")

# make model 
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])
#128, 64, 32, dense 128 - accuracy: 0.9225 - loss: 0.3222 - val_accuracy: 0.9213 - val_loss: 0.3193
def build_model(output_classes=10, input_shape=(28,28,1)):
    reg = regularizers.l2(l2=0.00005)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        #Conv Block 1
        tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        #Conv Block 2
        tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        #Conv Block 3
        tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        #Dense layers
        tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=reg),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.25),
        #Output layer
        tf.keras.layers.Dense(output_classes, activation='softmax')
    ])
    return model



def build_regression_model(input_shape=(28,28,1)):
    reg = regularizers.l2(l2=0.00005)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.02),
    ])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='tanh'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, use_bias=False, kernel_regularizer=reg, activation='tanh'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=reg, activation='tanh'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=reg, activation='tanh'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model

callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True), 
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001) 
             ]

# model_classification = build_model(output_classes=10)

# model_classification.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['accuracy']
# )

# classification_history = model_classification.fit(
#     train_images, train_labels,
#     epochs=45,
#     batch_size=128,
#     validation_split=0.1,
#     callbacks=callbacks
# )

# #Evaluate classification
# test_loss, test_acc = model_classification.evaluate(test_images, test_labels, verbose=2)
# print('\nClassification Test accuracy:', test_acc)

# classification_preds = np.argmax(model_classification.predict(test_images), axis=1)
# model_classification.summary()

#Start regression model
model_reg = build_regression_model(input_shape=(28,28,1))

model_reg.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mae']
)

reg_history = model_reg.fit(
    train_reg_images, train_labels,
    epochs=45,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks
)

#Evaluate regression
test_reg_loss, test_reg_mae = model_reg.evaluate(test_reg_images, test_labels, verbose=2)
print('\nRegression Test MAE:', test_reg_mae)

reg_preds = model_reg.predict(test_reg_images)
model_reg.summary()

#PLOTS

#Plot 1
#training_classification_curve = evaluate.plot_classification_learning_curve(classification_history, save_path="models/training_classification_curve.png")

#Plot 2
training_regression_curve = evaluate.plot_regression_learning_curve(reg_history, save_path="models/training_regression_curve.png")

#Plot 3
#confusion_matrix = evaluate.plot_confusion_matrix(test_labels, classification_preds, labels, normalize=True, save_path="models/confusion_matrix.png")

#Plot 4
#class_residuals_vs_predicted = evaluate.plot_residuals_vs_predicted(test_labels, classification_preds, save_path="models/classification_residuals_vs_predicted.png")
reg_residuals_vs_predicted = evaluate.plot_residuals_vs_predicted(test_labels, reg_preds, save_path="models/reg_residuals_vs_predicted.png")


#Plot 5
# Initialize JS visualization
shap.initjs()

# Use small subset for memory efficiency
background = train_images[:100]
test_subset = test_images[:10]

# explainer = shap.DeepExplainer(model_classification, background)
# shap_values = explainer.shap_values(test_subset)

# # Visualize global importance
# shap.image_plot(shap_values, test_subset)

# feature_names = [f"pixel_{i}_{j}" for i in range(28) for j in range(28)]
# shap.summary_plot(shap_values, feature_names)
# #Visualizing each feature's impact on decision using waterfall plot
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(), shap_values[0][0], feature_names)
# # visualize all the training set predictions
# shap.plots.force(explainer.expected_value.numpy(), shap_values[0], feature_names)

explainer_reg = shap.DeepExplainer(model_reg, train_reg_images[:100])
shap_values_reg = explainer_reg.shap_values(test_reg_images[:10])
shap.image_plot(shap_values_reg, test_reg_images[:10])

#Table 1
# class_f1 = evaluate.evaluate_precision_recall_f1(test_labels, classification_preds)
reg_f1 = evaluate.evaluate_precision_recall_f1(test_labels, reg_preds)

# #Table 2
# class_mae = evaluate.evaluate_regressive_model(test_labels, classification_preds)
# class_rmse = evaluate.evaluate_regression_rmse(test_labels, classification_preds)
# print({"F1 Score: {class_f1} MAE: {class_mae} RMSE: {class_rmse}"})

reg_mae = evaluate.evaluate_regressive_model(test_labels, reg_preds)
reg_rmse = evaluate.evaluate_regression_rmse(test_labels, reg_preds)

print({"F1 Score: {reg_f1} MAE: {reg_mae} RMSE: {reg_rmse}"})

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()


# ###methods taken from the notes
# # We can plot the history of the accuracy and see how it improves over time (epochs)
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.7, 1])
# plt.legend(loc='best')
# plt.show()

# # We can plot the history of the loss (cost function) and see how it decreases
# plt.plot(history.history['loss'], label='loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()
