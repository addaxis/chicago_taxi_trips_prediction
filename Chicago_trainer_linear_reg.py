import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from google.cloud import bigquery
import time

def read_data_from_bigquery():
    client = bigquery.Client()
    query = """
        SELECT * FROM aa-ai-specialisation.chicago_taxi_trips.processed_local_taxi_trips
    """
    df = client.query(query).to_dataframe()
    return df

def linear_regression_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=input_shape)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    df = read_data_from_bigquery()

    labels = df['fare'].values
    features = df.drop('fare', axis=1).values
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    model = linear_regression_model((X_train.shape[1],))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    EXPORT_PATH = 'gs://aa_chicago_taxi_trips/export/lr_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    model.save(EXPORT_PATH)


# import tensorflow as tf
# from sklearn.model_selection import train_test_split, KFold
# import pandas as pd
# import numpy as np
# from google.cloud import bigquery
# import time

# def read_data_from_bigquery():
#     client = bigquery.Client()
#     query = """
#         SELECT * FROM aa-ai-specialisation.chicago_taxi_trips.processed_local_taxi_trips
#     """
#     df = client.query(query).to_dataframe()
#     return df

# def linear_regression_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(1, input_shape=input_shape)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# from sklearn.metrics import mean_squared_error

# def cross_validate_model(X, y, n_splits=5):
#     best_mse = float('inf')
#     best_model = None
#     kf = KFold(n_splits=n_splits)
    
#     for train_index, val_index in kf.split(X):
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
#         X_train = np.asarray(X_train).astype('float32')
#         y_train = np.asarray(y_train).astype('float32')
#         X_val = np.asarray(X_val).astype('float32')
#         y_val = np.asarray(y_val).astype('float32')

#         model = linear_regression_model((X_train.shape[1],))
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#         model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])
        
#         y_pred = model.predict(X_val)
#         mse = mean_squared_error(y_val, y_pred)

#         if mse < best_mse:
#             best_mse = mse
#             best_model = model

#     return best_mse, best_model

# if __name__ == "__main__":
#     df = read_data_from_bigquery()

#     labels = df['fare'].values
#     features = df.drop('fare', axis=1).values
    
#     best_mse, best_model = cross_validate_model(features, labels, n_splits=5)
#     print(f"Best MSE: {best_mse}")

#     EXPORT_PATH = 'gs://aa_chicago_taxi_trips/export/lr_cv_best_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
#     best_model.save(EXPORT_PATH)

