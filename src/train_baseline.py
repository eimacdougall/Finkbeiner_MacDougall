import os
import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import data
import features
import evaluate

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def run_models(preprocess_method="minmax"): #Look at the features.py for options, default is minmax
    #Load data
    X_train, y_train = data.load_mnist('data/fashion', kind='train')
    X_test, y_test = data.load_mnist('data/fashion', kind='t10k')

    #Preprocess
    X_train_proc, scaler = features.preprocess(X_train, preprocess_method)
    X_test_proc = scaler.transform(X_test)

    #Save scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs")
    }

    #In the terminal run mlflow ui then follow the link
    #You might need to specifiy the port with --port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("FashionMNIST_Baselines") #Just naming this goober

    accuracies = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        #Evaluate
        metrics_dict = evaluate.evaluate_model(y_test, y_pred)
        print(f"{name} Accuracy: {metrics_dict['accuracy']:.4f}")
        accuracies.append(metrics_dict['accuracy'])

        #Save model in models // I wanted to have it like the mlartifacts folder which has everything but mlflow saves there automatically
        #idk how to fix that right now
        model_path = os.path.join("models", f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

        #Plot and save artifacts
        cm_path = os.path.join("models", f"{name.replace(' ', '_')}_confusion_matrix.png")
        evaluate.plot_confusion_matrix(y_test, y_pred, labels, title=f"{name} Confusion Matrix", save_path=cm_path)

        pred_path = os.path.join("models", f"{name.replace(' ', '_')}_predictions.png")
        evaluate.show_predictions(name, X_test, y_test, y_pred, labels, n=10, save_path=pred_path)

        if name == "Naive Bayes":
            means_path = os.path.join("models", f"{name.replace(' ', '_')}_feature_means.png")
            evaluate.plot_naive_bayes_means(model, labels, save_path=means_path)

        #Log everything to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("preprocess", preprocess_method)
            mlflow.log_metric("accuracy", metrics_dict["accuracy"])
            signature = infer_signature(X_train_proc, model.predict(X_train_proc))
            mlflow.sklearn.log_model(model, f"{name.replace(' ', '_')}_model", signature=signature)

            #Log plot artifacts
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(pred_path)
            if name == "Naive Bayes":
                mlflow.log_artifact(means_path)

    #Accuracy comparison plot
    acc_path = os.path.join("models", "model_accuracy_comparison.png")
    evaluate.plot_model_accuracies(list(models.keys()), accuracies, save_path=acc_path)
    mlflow.log_artifact(acc_path)

if __name__ == "__main__":
    run_models(preprocess_method="minmax")