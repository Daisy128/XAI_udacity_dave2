import os
import csv
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    root_path = '/home/jiaqq/Documents/ThirdEye-II/perturbationdrive/logs/mountain/'

    for root, dirs, files in os.walk(root_path, topdown=False):
        for file in files:
            if file.endswith('.csv'):
                csv = os.path.join(root, file)
                data_df = pd.read_csv(csv)

                if 'frameId' not in data_df.columns:
                    data_df.rename(columns={'index': 'frameId'}, inplace=True)
                    # data_df['frameId'] = data_df['image_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
                    data_df['index'] = range(1, len(data_df) + 1)

                    columns = ['index'] + [col for col in data_df.columns if col != 'index']
                    data_df = data_df[columns]

                    data_df.set_index('index', inplace=True)

                    data_df.to_csv(csv)
                    print("File: "+ csv + " has been modified")