#all da imports
import tensorflow as tf
from keras_tuner import RandomSearch
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import data
import os

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Load data
train_images, train_labels = data.load_mnist('data\\fashion', kind='train')
test_images, test_labels = data.load_mnist('data\\fashion', kind='t10k')

#fix data shape (ensure low values)
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"test shape: {test_images.shape}")

#ensure shape valid
train_images = train_images.reshape((-1, 28, 28))
test_images = test_images.reshape((-1, 28, 28))

#I think we were inputting the wrong shape and missing the channel dimension
#(60000, 28, 28) to (60000, 28, 28, 1) for greyscale
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)

print(f"test shape after reshape: {test_images.shape}")

# make model
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=hp.Choice('filters', [32, 64, 96]),
            kernel_size=3,
            activation='relu',
            padding='same',
            input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hp.Int('dense_units', min_value=64, max_value=256, step=64),
            activation='relu'
        ),
        tf.keras.layers.Dropout(
            hp.Float('dropout', 0.0, 0.5, step=0.1)
        ),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    #compile 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 0.0001, 0.01, sampling='log')), #the log should help find more dramatic different lr values
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    overwrite=True,
    directory='tuner_results',
    project_name='fashion_mnist',
    seed=42
)

#define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2)
]

#search for best hyperparameters and build the best model
tuner.search(train_images, train_labels, epochs=10, validation_split=0.1, batch_size=64, callbacks=callbacks)
best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hp)

#train on train data
best_model.fit(
    train_images, train_labels,
    epochs=10,
    validation_split=0.1,
    batch_size=64,
    callbacks=callbacks
)

# Evaluate the model with test data (no data leak)
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print("Best test accuracy:", test_acc, test_loss)

predictions = best_model.predict(test_images)
(predictions[0])
np.argmax(predictions[0])



# def plot_image(i, predictions_array, true_label, img):
#   """plot image and predictions"""
#   true_label, img = true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])

#   plt.imshow(img, cmap=plt.cm.binary)

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'

#   plt.xlabel("{} {:2.0f}% ({})".format(labels[predicted_label],
#                                 100*np.max(predictions_array),
#                                 labels[true_label]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   true_label = true_label[i]
#   plt.grid(False)
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')



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


###methods taken from the notes
# # We can plot the history of the accuracy and see how it improves over time (epochs)
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.7, 1])
# plt.legend(loc='best')
# plt.show()

# # We can plot the history of the loss (cost function) and see how it decreases
# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='best')
# plt.show()

# #Plot 1
# class_learning_curve_plot = os.path.join("models", "NN_FashionMNIST_Learning_Curve.png")
# evaluate.plot_classification_learning_curve(history, save_path=class_learning_curve_plot)

# #Plot 2
# regression_learning_curve_plot = os.path.join("models", "NN_FashionMNIST_Regression_Learning_Curve.png")
# evaluate.plot_regression_learning_curve(history, save_path=regression_learning_curve_plot)

# #Plot 3
confusion_matrix_plot = os.path.join("models", "NN_FashionMNIST_Confusion_Matrix.png")
evaluate.plot_confusion_matrix(test_labels, np.argmax(predictions, axis=1), labels, 
                                title="NN Fashion MNIST Confusion Matrix", save_path=confusion_matrix_plot)

# #Plot 4
# residuals_vs_predicted_plot = os.path.join("models", "NN_FashionMNIST_Residuals_vs_Predicted.png")
# evaluate.plot_residuals_vs_predicted(test_labels, np.argmax(predictions, axis=1), 
#                                     title="NN Fashion MNIST Residuals vs Predicted", save_path=residuals_vs_predicted_plot)

#Plot 5


#Table 1


#Table 2
