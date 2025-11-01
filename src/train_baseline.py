import os
import joblib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from sklearn.discriminant_analysis import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

import data
import features
import evaluate

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def run_models(preprocess_method="minmax"): #Look at the features.py for options, default is minmax
    #Load data
    X_train, y_train = data.load_mnist('data/fashion', kind='train')
    X_test, y_test = data.load_mnist('data/fashion', kind='t10k')

    X_regression_train = np.std(X_train, axis=1)
    X_regression_test = np.std(X_test, axis=1)

    #Preprocess
    X_train_proc, X_test_proc, scaler = features.preprocess(X_train, X_test, preprocess_method)
    #X_test_proc = scaler.transform(X_test)

    #Save scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    #Plots for EDA
    target_plot = os.path.join("models", "target_distribution.png")
    evaluate.plot_target_distribution(y_train, labels, save_path=target_plot)

    #Heatmap or boxplot of feature correlations
    #This shit takes way to long for all samples with heatmap
    #For boxplot it makes a grey blob because of the dataset size
    #Probably just reduce the amount of data being fed for this
    corr_plot = os.path.join("models", "feature_correlation_heatmap.png")
    #evaluate.plot_feature_correlations(X_train_proc, save_path=corr_plot)

    #std values for regression (contrast)
    X_train_reg_proc, scaler = features.regression_preprocess(X_train)
    X_test_reg_proc = scaler.transform(X_test)

    #Save scaler
    scaler_path = os.path.join("models", "regression_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    classification_models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=100, solver="lbfgs"),
    }
    regression_models = {
        "Linear Regression":Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
    ]),
        "Decision Tree Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", DecisionTreeRegressor()),
    ]),
    }

    #In the terminal run mlflow ui then follow the link
    #You might need to specifiy the port with --port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("FashionMNIST_Baselines") #Just naming this goober

    accuracies = []
    ##classification models
    for name, model in classification_models.items():
        print(f"\nTraining classification {name}...")
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
        
        #target distribution
        dist_path = os.path.join("models", f"{name.replace(' ', '_')}_target_distribution.png")
        evaluate.plot_target_distribution(y_pred, labels, title=f"{name} Target Distribution", save_path=dist_path)

        #confusion matrix
        cm_path = os.path.join("models", f"{name.replace(' ', '_')}_confusion_matrix.png")
        evaluate.plot_confusion_matrix(y_test, y_pred, labels, title=f"{name} Confusion Matrix", save_path=cm_path)

        #prediction examples
        pred_path = os.path.join("models", f"{name.replace(' ', '_')}_predictions.png")
        evaluate.show_predictions(name, X_test, y_test, y_pred, labels, n=10, save_path=pred_path)

        if name == "Naive Bayes":
            means_path = os.path.join("models", f"{name.replace(' ', '_')}_feature_means.png")
            evaluate.plot_naive_bayes_means(model, labels, save_path=means_path)

        #Not sure how useful this is for classification (Naive Bayes) but whatever
        residuals_path = os.path.join("models", f"{name.replace(' ', '_')}_residuals.png")
        evaluate.plot_residuals_vs_predicted(y_test, y_pred, save_path=residuals_path)

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
            mlflow.log_artifact(dist_path)
            if name == "Naive Bayes":
                mlflow.log_artifact(means_path)
            if residuals_path:
                mlflow.log_artifact(residuals_path)
            mlflow.end_run()
    #Accuracy comparison plot
    acc_path = os.path.join("models", "model_accuracy_comparison.png")
    evaluate.plot_model_accuracies(list(classification_models.keys()), accuracies, save_path=acc_path)
    mlflow.log_artifact(target_plot)
    #mlflow.log_artifact(corr_plot)
    mlflow.log_artifact(acc_path)
    mlflow.end_run()

    ##regression 
    for name, model in regression_models.items():
        print(f"\nTraining regression {name}...")
        model.fit(X_train_reg_proc, y_train)
        y_pred = model.predict(X_test_reg_proc)

        #Evaluate
        metrics_dict = evaluate.evaluate_regressive_model(y_test, y_pred)
        print(f"{name} MAE: {metrics_dict['MAE']:.4f}")
        accuracies.append(metrics_dict['MAE'])

        #Save model in models 
        #idk how to fix what ethan wanted to fix either
        model_path = os.path.join("models", f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

        #Plot and save artifacts
        cm_path = os.path.join("models", f"{name.replace(' ', '_')}_residuals.png")
        evaluate.plot_residuals_vs_predicted(y_test, y_pred, title=f"{name} residuals", save_path=cm_path)

        pred_path = os.path.join("models", f"{name.replace(' ', '_')}_predictions.png")
        evaluate.plot_regression_predictions(y_test, y_pred, save_path=pred_path)

        #Log everything to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("preprocess", preprocess_method)
            mlflow.log_metric("MAE", metrics_dict["MAE"])
            signature = infer_signature(X_train_reg_proc, model.predict(X_train_reg_proc))
            mlflow.sklearn.log_model(model, f"{name.replace(' ', '_')}_model", signature=signature)

            #Log plot artifacts
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(pred_path)
            mlflow.end_run()


import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    run_models(preprocess_method="none")
