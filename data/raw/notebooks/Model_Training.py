from exploration_and_peak_logic import train_scaled, test_scaled
import numpy as np
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