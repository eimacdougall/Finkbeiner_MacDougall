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
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
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

