#all da imports
import tensorflow as tf
from keras import regularizers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import data
import evaluate
import features
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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

def build_model(output_classes=10,
                input_shape=(28,28,1),
                dropout_rate=0.3,
                activation='relu',
                conv_filters=[64, 128, 256],
                conv_kernel_size=3,
                dense_units=[256, 128]):
    reg = regularizers.l2(l2=0.00005)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation
    ])
    
    # Convolutional blocks
    for filters in conv_filters:
        model.add(tf.keras.layers.Conv2D(filters, (conv_kernel_size, conv_kernel_size),
                                         padding='same', use_bias=False, kernel_regularizer=reg))
        model.add(tf.keras.layers.Activation(activation))
        model.add(tf.keras.layers.Conv2D(filters, (conv_kernel_size, conv_kernel_size),
                                         padding='same', use_bias=False, kernel_regularizer=reg))
        model.add(tf.keras.layers.Activation(activation))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2,2)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    
    # Dense layers
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=reg))
        model.add(tf.keras.layers.Activation(activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output
    model.add(tf.keras.layers.Dense(output_classes, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_classification_model(output_classes=10, input_shape=(28,28,1)):
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
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(128, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(128, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(256, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Conv2D(256, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),

        tf.keras.layers.Conv2D(256, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    return model

callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), 
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001) 
             ]

model_classification = build_classification_model(output_classes=10)

model_classification.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

classification_history = model_classification.fit(
    train_images, train_labels,
    epochs=45,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks
)

# #Evaluate classification
test_loss, test_acc = model_classification.evaluate(test_images, test_labels, verbose=2)
print('\nClassification Test accuracy:', test_acc)

classification_preds = np.argmax(model_classification.predict(test_images), axis=1)
model_classification.summary()

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
training_classification_curve = evaluate.plot_classification_learning_curve(classification_history, save_path="models/training_classification_curve.png")

#Plot 2
training_regression_curve = evaluate.plot_regression_learning_curve(reg_history, save_path="models/training_regression_curve.png")

#Plot 3
confusion_matrix = evaluate.plot_confusion_matrix(test_labels, classification_preds, labels, normalize=False, save_path="models/confusion_matrix.png")

#Plot 4
class_residuals_vs_predicted = evaluate.plot_residuals_vs_predicted(test_labels, classification_preds, save_path="models/classification_residuals_vs_predicted.png")
reg_residuals_vs_predicted = evaluate.plot_residuals_vs_predicted(test_labels, reg_preds, save_path="models/reg_residuals_vs_predicted.png")

#Plot 5
dropout_rates = [0.0, 0.1, 0.2, 0.3]
values, accuracies = evaluate.run_ablation_study(
    create_model_func=build_model, 
    x_train=train_images,
    y_train=train_labels,
    x_test=test_images,
    y_test=test_labels,
    param_name='dropout_rate',
    param_values=dropout_rates,
    epochs=5
)
evaluate.plot_ablation(values, accuracies, 'dropout_rate', save_path="models/ablation_dropout_rate.png")

filter_configs = [
    [32, 64, 128],
    [64, 128, 256],
    [128, 256, 512]
]
values, accuracies = evaluate.run_ablation_study(
    create_model_func=build_model,
    x_train=train_images,
    y_train=train_labels,
    x_test=test_images,
    y_test=test_labels,
    param_name='conv_filters',
    param_values=filter_configs,
    epochs=5
)
evaluate.plot_ablation(values, accuracies, 'Conv Filter Counts', save_path="models/ablation_filter_counts.png")

dense_configs = [
    [128, 64],
    [256, 128],
    [512, 256]
]
values, accuracies = evaluate.run_ablation_study(
    create_model_func=build_model,
    x_train=train_images,
    y_train=train_labels,
    x_test=test_images,
    y_test=test_labels,
    param_name='dense_units',
    param_values=dense_configs,
    epochs=5
)
evaluate.plot_ablation(values, accuracies, 'Dense Layer Units', save_path="models/ablation_dense_units.png")

# Activation ablation
activations = ['relu', 'sigmoid', 'tanh', 'linear']
values, acc = evaluate.run_ablation_study(
    create_model_func=build_model,
    x_train=train_images,
    y_train=train_labels,
    x_test=test_images,
    y_test=test_labels,
    param_name='activation',
    param_values=activations,
    epochs=5
)
evaluate.plot_ablation(values, acc, 'activation', save_path="models/ablation_activation_function.png")

kernel_sizes = [1, 3, 5]
values, accuracies = evaluate.run_ablation_study(
    create_model_func=build_model,
    x_train=train_images,
    y_train=train_labels,
    x_test=test_images,
    y_test=test_labels,
    param_name='conv_kernel_size',
    param_values=kernel_sizes,
    epochs=5
)
evaluate.plot_ablation(values, accuracies, 'Conv Kernel Size', save_path="models/ablation_kernel_size.png")

# Input column ablation
input_ablation_scores = []
input_slices = list(range(0, 28, 4))
for i in input_slices:
    x_train_ablated = np.copy(train_images)
    x_test_ablated = np.copy(test_images)
    x_train_ablated[:, :, i:min(i+4,28), :] = 0
    x_test_ablated[:, :, i:min(i+4,28), :] = 0

    model = build_model()
    model.fit(x_train_ablated, train_labels, epochs=5, validation_split=0.1, verbose=0)
    loss, acc = model.evaluate(x_test_ablated, test_labels, verbose=0)
    input_ablation_scores.append(acc)

evaluate.plot_ablation(input_slices, input_ablation_scores, 'pixel columns', save_path="models/ablation_input_pixel_columns.png")

#TABLES 

#Table 1
class_f1 = evaluate.evaluate_precision_recall_f1(test_labels, classification_preds)
class_mae = evaluate.evaluate_regressive_model(test_labels, classification_preds)
class_rmse = evaluate.evaluate_regression_rmse(test_labels, classification_preds)

print({"Classification F1": class_f1, "MAE": class_mae, "RMSE": class_rmse})

#Table 2
reg_mae = evaluate.evaluate_regressive_model(test_labels, reg_preds)
reg_rmse = evaluate.evaluate_regression_rmse(test_labels, reg_preds)

print({"Regression MAE": reg_mae, "RMSE": reg_rmse})