from exploration_and_peak_logic import train_scaled, test_scaled
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.integration.keras import WandbCallback
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv() # This loads the variables from the .env file
wandb.login(key=os.getenv("WANDB_API_KEY"))



window_size=96
def create_sequences(data,window_size):
    X,y=[],[]

    for i in range(window_size,len(data) ):
        
        X.append(data[i-window_size:i,:])  #all features except the target variable
        y.append(data[i,0])   #target variable is the last column (Usage_kWh)
        
       
    return np.array(X),np.array(y)
X_train,y_train=create_sequences(train_scaled,window_size)
X_test,y_test=create_sequences(test_scaled,window_size)
#for testing purpose only
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1)
])

wandb.init(project="Industrial Power Optimizer",
           name="LSTM_model",
           config={
               'lstm_units_1': 64,
               'lstm_units_2': 32,
               'dropout': 0.2,
               'batch_size': 32,
               'epochs': 50,
               'window_size':96,
               'optimizer': 'adam',
                         }
                         ,reinit=True)
early_stop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
wandb_callback = WandbCallback()


model.compile(optimizer='adam',loss='mse',metrics=['mae','mape'])
model.summary()