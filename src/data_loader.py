from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os

    # fetch dataset 
def get_raw_data():
        steel_industry_energy_consumption = fetch_ucirepo(id=851) 
        
        # data (as pandas dataframes) 
        df_raw=pd.DataFrame(steel_industry_energy_consumption.data.original)
         
        
        # metadata 
        #print(steel_industry_energy_consumption.metadata) 
        
        # variable information 
        
        #print(steel_industry_energy_consumption.variables)
        print(df_raw.head())
        
        
        #df_raw=pd.concat([X,y],axis=1)
        df_raw.drop_duplicates(inplace=True)
        os.makedirs("data/raw",exist_ok=True)
        df_raw.to_csv("data/raw/steel_industry_energy_consumption.csv",index=False)
        return df_raw


if __name__=="__main__":
    df=get_raw_data()
    print("Data loaded successfully")
