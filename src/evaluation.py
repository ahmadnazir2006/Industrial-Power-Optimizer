import joblib
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
model=load_model('models/industrial_lstm_model.keras.h5')
scaler=joblib.load('models/scaler.pkl')

test_scaled=np.load("data/raw/processed/test_scaled.npy")
df=pd.read_csv("data/raw/processed/steel_industry_final.csv",parse_dates=['date'],index_col='date')






def create_sequences(data, window_size=96):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :]) # Use all 7 features
        y.append(data[i, 0])              # Target is Usage_kWh
    return np.array(X), np.array(y)

X_test_3d, y_test_labels = create_sequences(test_scaled, 96)


y_pred=model.predict(X_test_3d)


def inverse_transform(data, scaler):
    dummy_pred=np.zeros((len(data),7))
    dummy_pred[:,0]=data.flatten()  #this is scaled version of the data
    actual=scaler.inverse_transform(dummy_pred)
    return actual[:,0]  #this is the actual values

y_pred_actual=inverse_transform(y_pred, scaler)
y_test_actual=inverse_transform(y_test_labels, scaler)


split_index=int(len(df)*0.8)
df_test=df.iloc[split_index+96:]  #the test set starts from the index after the last training point (which is split_index) and we need to add 96 to account for the window size
y_test_actual=y_test_actual[:len(df_test)]  #aligning the length of y_test_actual with df_test
y_pred_actual=y_pred_actual[:len(df_test)]  #aligning the length of y_pred_actual with df_test

plot_data=pd.DataFrame({'Actual Usage (KWh)': y_test_actual, 'Predicted Usage (KWh)': y_pred_actual}, index=df_test.index)
plot_data.to_csv("data/raw/processed/prediction_results.csv",index='date')
final_week=plot_data.loc['25-12-2018 00:00:00':'31-12-2018 23:45:00']
plt.figure(figsize=(15,6))

plt.plot(final_week.index, final_week['Predicted Usage (KWh)'], color='blue', linestyle='-',label='AI Predicted Usage (KWh)')
plt.plot(final_week.index, final_week['Actual Usage (KWh)'], color='orange', linestyle='--',label='Actual Usage (KWh)')

plt.xlabel('Date and Time')
plt.ylabel('Electicity Usage (KWh)')
plt.title('Industrial Power Optimizer: Model Performance (Final Week of 2018)')
plt.legend()
plt.show()


mask=plot_data['Actual Usage (KWh)']>1
MAPE=np.mean(np.abs(plot_data['Actual Usage (KWh)'][mask]-plot_data['Predicted Usage (KWh)'][mask])/plot_data['Actual Usage (KWh)'][mask])
print(f"MAPE: {MAPE:.2f}")
Accuracy=100-MAPE*100
print(f"Accuracy: {Accuracy:.2f}%")

