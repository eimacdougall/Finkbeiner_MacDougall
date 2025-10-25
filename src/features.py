from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def preprocess(X_train, preprocess_method="minmax"):
    #Apply scaling/normalization and return transformed sets
    if preprocess_method == "standard":
        scaler = StandardScaler()
    elif preprocess_method == "normalize":
        scaler = Normalizer()
    else:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler