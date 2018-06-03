import pandas as pd

root_path = '/home/storage/lsy/fashion/'

dfs = []
for cloth in ['blouse', 'skirt', 'outwear', 'dress', 'trousers']:
    df = pd.read_csv(root_path + 'kp_predictions/' + cloth + '.csv')
    dfs.append(df)
res_df = pd.concat(dfs)
res_df.to_csv(root_path+'kp_predictions/result.csv', index=False)