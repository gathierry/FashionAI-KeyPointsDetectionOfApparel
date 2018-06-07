import pandas as pd
from src.config import Config

config = Config()

dfs = []
for cloth in ['blouse', 'skirt', 'outwear', 'dress', 'trousers']:
    df = pd.read_csv(config.proj_path + 'kp_predictions/' + cloth + '.csv')
    dfs.append(df)
res_df = pd.concat(dfs)
res_df.to_csv(config.proj_path +'kp_predictions/result.csv', index=False)