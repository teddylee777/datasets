import pandas as pd

# 데이터셋 JSON 파일 경로
DATASET_URL = 'https://raw.githubusercontent.com/teddylee777/datasets/master/teddynote/data/dataset.csv'


def list_all():
    return pd.read_csv(DATASET_URL)

def load(name):
    ds = pd.read_csv(DATASET_URL)
    row = ds.loc[ds['data'] == name]
    if len(row) > 0:
        row['data'], row['description'],


if __name__ == '__main__':
    print(list_all())

