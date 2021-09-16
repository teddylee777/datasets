from tqdm.notebook import tqdm
import pandas as pd
import requests
import os
import zipfile

# 데이터셋 파일 경로
# DATASET_URL = 'https://raw.githubusercontent.com/teddylee777/datasets/master/teddynote/data/dataset.csv'
# METADATA_URL = 'https://raw.githubusercontent.com/teddylee777/datasets/master/teddynote/data/metadata.csv'

# 데이터셋 파일 경로 (dropbox 림크)
DATASET_URL = 'https://www.dropbox.com/s/95wzfrmoc4qrfvw/dataset.csv?dl=1'
METADATA_URL = 'https://www.dropbox.com/s/9y1nyvwy95jh2w3/metadata.csv?dl=1'


class Dataset():
    def __init__(self, name, desc, _id, data_dir='data'):
        self.name = name
        self.desc = desc
        self.data_dir = data_dir
        self.id = _id

        # data 폴더 생성
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.load_metadata()

    def load_metadata(self):
        metadata = pd.read_csv(METADATA_URL)
        self.meta = {i: v for i, v in metadata.loc[metadata['id'] == self.id, ['filename', 'url']].values}

    def download(self):
        print('======= 다운로드 시작 =======\n')
        for idx, (filename, url) in enumerate(self.meta.items()):
            r = requests.get(url, stream=True)
            filepath = os.path.join(self.data_dir, filename)

            ## 다운로드 progress bar 추가 ##
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024

            print(f'{filepath}')

            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)

            with open(filepath, 'wb') as file:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR: 다운로드 도중 에러가 발생하였습니다.")
            else:
                if filepath.endswith('.zip'):
                    print(f'압축 해제 및 프로젝트 파일 구성중...')
                    zipfile.ZipFile(filepath).extractall(os.path.join(self.data_dir, project_name))

        print('\n======= 다운로드 완료 =======')


def list_all():
    ret = pd.read_csv(DATASET_URL).iloc[:, :-1]
    ret.columns = ['데이터셋', '설명']
    return ret


def download(name, data_dir='data'):
    ds = pd.read_csv(DATASET_URL)
    row = ds.loc[ds['data'] == name]
    if len(row) > 0:
        dataset = Dataset(row['data'], row['data_desc'], row['id'].iloc[0], data_dir=data_dir)
        dataset.download()