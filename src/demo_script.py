#all da imports
import tensorflow as tf
from keras import regularizers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import data
from features import regression_preprocess
import random
import mlflow.tensorflow
import mlflow
from evaluate import plot_confusion_matrix, plot_classification_learning_curve, show_predictions

#set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
    
#define model ID
model_id = "d7362372d8fa437db109c8afd95c8c81"

#Load data
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images_raw, train_labels = data.load_mnist('data\\fashion', kind='train')
test_images_raw, test_labels = data.load_mnist('data\\fashion', kind='t10k')

#fix data shape (ensure low values)
train_images = train_images_raw / 255.0
test_images = test_images_raw / 255.0


print(f"test shape: {test_images.shape}")

#ensure shape valid
train_images = train_images.reshape((-1, 28, 28))
test_images = test_images.reshape((-1, 28, 28))

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

print(f"test shape after reshape: {test_images.shape}")
print(f"train shape after reshape: {train_images.shape}")

# make model 
import mlflow.keras 

model_uri = f"runs:/{model_id}/model"
model = mlflow.keras.load_model(model_uri)

#evaluate
model.summary()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nClassification Test accuracy:', test_acc)

predicted = np.argmax(model.predict(test_images), axis=1)
confusion_matrix = plot_confusion_matrix(test_labels, predicted, labels, normalize=False, save_path="models/confusion_matrix.png")

model_name = "classification network"
show_predictions(model_name, test_images_raw, test_labels, predicted, labels)