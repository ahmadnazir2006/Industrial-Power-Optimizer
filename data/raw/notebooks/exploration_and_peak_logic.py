import sys ,os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.data_loader import get_raw_data
df=get_raw_data()
df['date']=pd.to_datetime(df['date'],errors='coerce', dayfirst=True)
df.set_index('date',inplace=True)
df.sort_index(inplace=True)
gap_required=pd.Timedelta("15min")

print(((df.index.to_series()).diff()).value_counts())
diffs=((df.index.to_series()).diff()).value_counts()
print("Actual gap between consecutive timestamps:")
print(diffs)
print(diffs[gap_required!=diffs].sum())


print(df.index.min(), df.index.max())
expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="15min")
print(len(expected), len(df))
missing = expected.difference(df.index)
print("Missing timestamps:", len(missing))



#defining threshold and creating a new column to identify peaks
threshold=df['Usage_kWh'].quantile(0.95)
df['is_peak']=(df['Usage_kWh']>threshold).astype(int)
print(df['is_peak'].value_counts())
X=df['01/01/2018 00:15':'07/01/2018 00:00']

# plt.figure(figsize=(12, 6))
# plt.plot(X.index,X['Usage_kWh'], linestyle='-')
# plt.axhline(threshold,label=f'Threshold: {threshold:.2f}', linestyle='--',color='red')
# peaks=X[X['Usage_kWh']>threshold]
# plt.scatter(peaks.index, peaks['Usage_kWh'], color='red', s=50, label='Peaks')
# plt.xlabel('Date and Time')
# plt.ylabel('Usage (kWh)')
# plt.title('Power Usage Over Time')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
# Apply map to the column
df['Day_num'] = df['Day_of_week'].map(day_map)

# 2. Calculate Cyclical Features using the new numeric column
df['NSM_sin'] = np.sin(2 * np.pi * (df['NSM'] / 86400))
df['NSM_cos'] = np.cos(2 * np.pi * (df['NSM'] / 86400))
df['Day_of_week_sin'] = np.sin(2 * np.pi * (df['Day_num'] / 7))
df['Day_of_week_cos'] = np.cos(2 * np.pi * (df['Day_num'] / 7))

# 3. Add Usage_kWh to features so we can see how time affects power!
features = ['Usage_kWh', 'NSM_sin', 'NSM_cos', 'Day_of_week_sin', 'Day_of_week_cos']

# Select features and ensure they are numeric
subset = df[features].copy()
for col in features:
    subset[col] = pd.to_numeric(subset[col], errors='coerce')

# Drop only rows that failed (there shouldn't be any now)
subset = subset.dropna()

print("Subset shape:", subset.shape)

# if subset.empty:
#     print("ERROR: No data available for heatmap - check your column names or mapping")
# else:
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         subset.corr(),
#         annot=True,     # Shows the correlation numbers
#         cmap='coolwarm', # Red for high correlation, blue for low
#         vmin=-1,
#         vmax=1,
#         fmt=".2f"
#     )
#     plt.title('Correlation Heatmap: Time vs Power Usage')
#     plt.tight_layout()
#     plt.show()

   
#using Binary encoding for WeekStatus column
df['WeekStatus'] = (
    df['WeekStatus']
    .astype(str)
    .str.strip()
    .str.lower()
    .map({'weekday': 0, 'weekend': 1})
) #1 for weekend, 0 for weekday
#print(df['WeekStatus'].unique())

#processed version of the data being stored
os.makedirs("data/raw/processed", exist_ok=True)
df.to_csv("data/raw/processed/steel_industry_processed.csv", index=True)


#selecting only the features we want to use for modeling
features_list = [
    'Usage_kWh', 
    'Lagging_Current_Reactive.Power_kVarh', 
    'NSM_sin', 'NSM_cos', 
    'Day_of_week_sin', 'Day_of_week_cos', 
    'WeekStatus'
]
#final dataset with only the features we want to use for modeling
df_final=df[features_list].copy()

os.makedirs("data/raw/processed", exist_ok=True)
df_final.to_csv("data/raw/processed/steel_industry_final.csv", index=True)

#finding how many rows to split for 80-20 train-test split in choronological order
print(df_final.value_counts().sum())
split_threshold=int(len(df_final)*0.8)
print(split_threshold)              #Total:35040  Train:28032  Test:7008
df_final_train=df_final.iloc[:split_threshold]
df_final_test=df_final.iloc[split_threshold:]

#for testing purpose only
print(f"Training ends at: {df_final_train.index.max()}")
print(f"Testing starts at: {df_final_test.index.min()}")

normalizer=MinMaxScaler()
#fitting only on train data to prevent data leakage
train_scaled=normalizer.fit_transform(df_final_train)   
#transforming test data using the same scaler fitted on train data
test_scaled=normalizer.transform(df_final_test)
#This is for testing purpose only
print(f"Train Scaled Min: {train_scaled.min()}")
print(f"Train Scaled Max: {train_scaled.max()}")
print(f"Test Scaled Min: {test_scaled.min()}")
print(f"Test Scaled Max: {test_scaled.max()}")
