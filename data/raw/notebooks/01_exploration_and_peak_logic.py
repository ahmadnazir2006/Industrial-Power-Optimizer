import sys ,os
import pandas as pd
from pathlib import Path
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