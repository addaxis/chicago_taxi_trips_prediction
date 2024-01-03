import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from google.cloud import bigquery
import time
from tensorflow import keras
from keras_tuner.tuners import RandomSearch

def read_data_from_bigquery():
    client = bigquery.Client()
    query = """
        SELECT * FROM aa-ai-specialisation.chicago_taxi_trips.processed_local_taxi_trips
    """
    df = client.query(query).to_dataframe()
    return df


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu',
        input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error')
    return model

if __name__ == "__main__":
    df = read_data_from_bigquery()

    labels = df['fare'].values
    
    features = df.drop('fare', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=features['trip_start_hour'])
    X_train = np.asarray(X_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        tf.keras.layers.Dense(1)
    ])
    learning_rate = 2.5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    best_model = keras.models.load_model('best_model.h5')
    # Plotting the training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('training_validation_loss.png')
        
    y_pred = model.predict(X_test)
    
    # MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")

    EXPORT_PATH = 'gs://aa_chicago_taxi_trips/export/model_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    model.save(EXPORT_PATH)
    LOCAL_PATH = 'model_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    model.save(LOCAL_PATH)
    BEST_LOCAL_PATH = 'best_model_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    best_model.save(BEST_LOCAL_PATH)
