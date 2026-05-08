import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import wandb
from wandb.integration.keras import WandbMetricsLogger
import numpy as np
import os
from dotenv import load_dotenv
import joblib
load_dotenv() # This loads the variables from the .env file
wandb.login(key=os.getenv("WANDB_API_KEY"))


train_scaled=np.load("data/raw/processed/train_scaled.npy")
test_scaled=np.load("data/raw/processed/test_scaled.npy")
scaler=joblib.load("models/scaler.pkl")

window_size=96
def create_sequences(data,window_size):
    X,y=[],[]

    for i in range(window_size,len(data)-window_size ):
        
        X.append(data[i-window_size:i,:])  #all features except the target variable
        y.append(data[i:i+window_size,0])   #target variable is the first column (Usage_kWh)
        
       
    return np.array(X),np.array(y)
X_train,y_train=create_sequences(train_scaled,window_size)
X_test,y_test=create_sequences(test_scaled,window_size)
#for testing purpose only
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

os.makedirs("models",exist_ok=True)
print("Starting model training...")

model_lstm=tf.keras.Sequential([
    
    tf.keras.layers.LSTM(64,return_sequences=True, input_shape=(96, 7)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(96)
])

model_cnn_lstm=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96,7)),
    tf.keras.layers.Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=(96,7)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64,return_sequences=False,input_shape=(96, 7)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(96)
])

model_gru=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(96,7)),
    tf.keras.layers.GRU(64,return_sequences=True, input_shape=(96, 7)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(32,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(96)
])

Model_dict={'LSTM':model_lstm,'CNN_LSTM':model_cnn_lstm,'GRU':model_gru}

for Model_name,model in Model_dict.items():
            tf.keras.backend.clear_session()
            wandb.init(project="Industrial Power Optimizer",
                    name=Model_name,
                    config={
                        'lstm_units_1': 64,
                        'lstm_units_2': 32,
                        'gru_units_1': 64,
                        'gru_units_2': 32,
                        'cnn_filters': 64,
                        'cnn_kernel_size': 3,
                        'dropout': 0.2,
                        'batch_size': 32,
                        'epochs': 50,
                        'window_size':96,
                        'optimizer': 'adam',
                                    }
                                    ,reinit=True)
            early_stop=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
            wandb_callback = WandbMetricsLogger()

            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
            loss_function=tf.keras.losses.Huber(delta=1.0)

            reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.5,
                                              patience=3,
                                              min_lr=0.00001)

            model.compile(optimizer=optimizer,loss=loss_function,metrics=['mae','mape'])
            history=model.fit(X_train,y_train,validation_split=0.1,epochs=wandb.config.epochs,batch_size=wandb.config.batch_size,callbacks=[early_stop,wandb_callback,reduce_lr],verbose=1)
            model.summary()
            # 1. Create the folder (if it's not already there)
            if not os.path.exists('models'):
                os.makedirs('models')

            # 2. Save the LSTM Model
            model.save(f'models/{Model_name.lower()}.h5')

            # 3. Save the Scaler (This is vital for Phase 5: Evaluation)
            joblib.dump(scaler, f'models/{Model_name.lower()}_scaler.pkl')