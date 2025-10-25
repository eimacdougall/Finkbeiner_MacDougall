### test scripts to use to understand everything ###
#use this for each image to see it
def print_image(x):
    for i in range(28):
        print(x[i*28:(i+1)*28])

#use this to print a comparison
def compare_predicted(pred, actual):
    if pred == actual:
        print("OK", labels[pred], labels[actual])
    else:
        print("WRONG", labels[pred], labels[actual])

#imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mnist_reader
import numpy as np
import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from sklearn import metrics


#get the data
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(X_train.shape
      ,y_train.shape
      ,X_test.shape
      ,y_test.shape)
print_image(X_train.tolist()[0]) ##prints the first image to the console
print(labels[y_train.tolist()[0]]) #prints the corresponding label

#preprocessing

scaler = StandardScaler()
normalizer = Normalizer()
minmax = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_normalized = normalizer.fit_transform(X_train)
X_train_minmax = minmax.fit_transform(X_train)

################################################################
#classifcation model tests/examples <get accuracy/F1
    #naieve bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnd_y_pred_class = mnb.predict(X_test)
print("-->Naive Bayes")
#for i in range(100): compare_predicted(mnd_y_pred_class[i], y_test[i])
    # metrics
print("-->Naieve Bayes Metrics")
print(metrics.accuracy_score(y_test, mnd_y_pred_class))
X= metrics.confusion_matrix(y_test, mnd_y_pred_class)
print(X)
        ##################
    #multinomial logistic
mlog = LogisticRegression(max_iter=2000, solver="lbfgs")
mlog.fit(X_train, y_train)
mlog_y_pred_class = mlog.predict(X_test)
compare_predicted(mlog_y_pred_class[0], y_test[0])
print("-->Multinomial Logistic")
#for i in range(100): compare_predicted(mlog_y_pred_class[i], y_test[i])
    # metrics
print("-->Multinomial Logistic Metrics")
print(metrics.accuracy_score(y_test, mlog_y_pred_class))
X= metrics.confusion_matrix(y_test, mlog_y_pred_class)
print(X)

print("-->Comparing Naive Bayes to Multinomial Logistic")
#for i in range(10):compare_predicted(mlog_y_pred_class[i], mnd_y_pred_class[i])
metrics.confusion_matrix(mlog_y_pred_class, mnd_y_pred_class)
        #####################
lr_deg3 = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler(with_mean=False)),
    ("logistic", LogisticRegression(max_iter=2000, solver="lbfgs"))
])

#######################################################################
#regression model tests/examples <get RMSE/MAE
    # preprosess data bc we want std of each 
X_std = StandardScaler().fit_transform(X_train)

    #linear regression
        #okay i have zero idea how to interpret this but it works (runs and returns a number lmao)

reg = LinearRegression().fit(X_std, y_train)
print(reg.score(X_std, y_train))
print("-->Linear Regression Coefficients")
print(reg.coef_)
print("-->Linear Regression Intercept")
print(reg.intercept_)
print("-->Linear Regression Prediction vs Actual")
print(reg.predict(X_std[0:2]))
print(y_train[0:2])
        
    #decision tree
detree = DecisionTreeRegressor().fit(X_std, y_train)
print("-->Decision Tree Regression Score")
print(detree.score(X_std, y_train))
print("-->Decision Tree Regression Prediction vs Actual")
print(detree.predict(X_std[0:2]))
print(y_train[0:2])


#idk figure out what the to do with matplotlib 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def show_predictions(model_name, X, y_true, y_pred, n=10):
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        color = "green" if y_true[i] == y_pred[i] else "red"
        plt.title(f"{labels[y_pred[i]]}\n({labels[y_true[i]]})", color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle(f"{model_name} Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    plt.show()

# Example usage
show_predictions("Naive Bayes", X_test, y_test, mnd_y_pred_class)
show_predictions("Logistic Regression", X_test, y_test, mlog_y_pred_class)

# Naive Bayes confusion matrix
plt.figure(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, mnd_y_pred_class, display_labels=labels, cmap="Blues", xticks_rotation=45)
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Logistic Regression confusion matrix
plt.figure(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(y_test, mlog_y_pred_class, display_labels=labels, cmap="Greens", xticks_rotation=45)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

accuracies = [
    metrics.accuracy_score(y_test, mnd_y_pred_class),
    metrics.accuracy_score(y_test, mlog_y_pred_class)
]
models = ['Naive Bayes', 'Logistic Regression']

plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
plt.ylim(0, 1)
plt.show()

import numpy as np

conf_mat = metrics.confusion_matrix(y_test, mnd_y_pred_class)
np.fill_diagonal(conf_mat, 0)
most_confused = np.unravel_index(conf_mat.argmax(), conf_mat.shape)
print(f"Naive Bayes most confused: {labels[most_confused[0]]} vs {labels[most_confused[1]]}")

means = np.exp(mnb.feature_log_prob_).reshape(10, 28, 28)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(means[i], cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.suptitle("Naive Bayes Learned Feature Means per Class", fontsize=14)
plt.show()
