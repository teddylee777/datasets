import codecs
import json
import os

# 코드 입력 문구 (코드를 삭제하고 출력을 박제하는 Cell)
code_input_msgs = [
    '# 코드입력',
    '# 코드를 입력하세요.',
    '# 코드를 입력해 주세요',
]

# 검증코드 (출력 값을 삭제하지 않음)
validation_msgs =  [
    '# 검증코드',
    '# 코드 검증',
    '# 코드검증',
]


def convert_ipynb(from_file, to_file=None, folder_path=None, post_fix='-변환.ipynb', remain_code=False):
    """
    from_file: 읽어 올 해설 파일 이름
    to_file: 변환 후 내보낼 파일 명, None
    folder_path: 기본 값 None. None이 아니면 해당 폴더경로에 생성
    post_fix: 파일 뒤에 붙혀줌. 그대로 두면 -변환 이라고 postfix 가 붙어서 자동 생성
    remain_code: default: False. True 적용시 # 코드입력 Cell의 코드를 살려둠
    
    (예시)
    - 폴더 지정 안하는 경우, 같은 경로에 생성
    convert_ipynb(filename)
    - 폴더 지정시 해당 경로의 폴더에 생성
    convert_ipynb(filename, folder_path='00-Workshop/변환')
    - 아무 post_fix 없이 생성
    convert_ipynb(filename, folder_path='00-Workshop/변환', post_fix='.ipynb')
    """
    global code_input_msgs, validation_msgs

    try:
        f = codecs.open(from_file, 'r')
        source = f.read()
    except UnicodeDecodeError:
        f = codecs.open(from_file, 'r', encoding='utf-8')
        source = f.read()
    except Exception as e:
        raise Exception("파일 변환에 실패 했습니다. 에러 메세지:" + e)

    # json 로드
    y = json.loads(source)

    idx = []
    sources = []

    for i, x in enumerate(y['cells']):
        flag = False
        valid_flag = False
        for x2 in x['source']:
            for msg in code_input_msgs:
                if msg in x2:
                    flag = True
                    break

            for valid_msg in validation_msgs:
                if valid_msg in x2:
                    valid_flag = True
                    break

        if flag and not remain_code:
            new_text = []
            for x2 in x['source']:
                if x2.startswith('#'):
                    new_text.append(x2)
            x['source'] = new_text

        if 'outputs' in x.keys():
            if not flag and not valid_flag:
                x['outputs'] = []  
            elif len(x['outputs']) > 0:
                for outputs in x['outputs']:
                    if 'data' in outputs.keys():
                        clear_flag = False
                        for key, value in outputs.items():
                            if type(value) == dict and len(value) > 0:
                                add_cnt = 0
                                for key2 in value.keys():
                                    if 'text/html' == key2:
                                        idx.append(i + add_cnt)
                                        html_text = value['text/html']
                                        html_text.insert(0, '<p><strong>[출력 결과]</strong></p>')
                                        sources.append(html_text)
                                        clear_flag = True
                                        break
                                    elif 'text/plain' == key2:
                                        plain_text = value['text/plain']
                                        if len(plain_text) > 0 and plain_text[0].startswith('<Figure'):
                                            continue
                                        idx.append(i + add_cnt)
                                        plain_text[0] = '<pre>' + plain_text[0]
                                        plain_text[len(plain_text)-1] = plain_text[len(plain_text)-1] + '</pre>'
                                        plain_text.insert(0, '<p><strong>[출력 결과]</strong></p>')
                                        sources.append(plain_text)
                                        clear_flag = True
                                        add_cnt += 1
                                    elif 'image/png' == key2:
                                        idx.append(i + add_cnt)
                                        plain_image = value['image/png']
                                        plain_image = '<img src="data:image/png;base64,' + plain_image.replace('\n','') + '"/>'
                                        sources.append(plain_image)
                                        clear_flag = True
                                        add_cnt += 1
                        if clear_flag:
                            x['outputs'] = []

                if len(x['outputs']) > 0 and 'text' in x['outputs'][0].keys():
                    idx.append(i)
                    text = x['outputs'][0]['text']
                    
                    if len(text) > 0:
                        text[0] = '<pre>' + text[0]
                        text[len(text) - 1] = text[len(text) - 1] + '</pre>'
                        text.insert(0, '<p><strong>[출력 결과]</strong></p>')
                    sources.append(text)
                    x['outputs'][0]['text'] = []

        if 'execution_count' in x.keys():
            x['execution_count'] = None

    cnt = 0
    tmp = []
    for i, s in zip(idx, sources):
        v = {'cell_type': 'markdown',
             'metadata': {},
             'source': s}
        tmp.append((i + 1 + cnt, v))
        cnt += 1

    for i in range(len(tmp)):
        y['cells'].insert(tmp[i][0], tmp[i][1])

    if to_file is None:
        if '해설' in from_file:
            to_file = from_file.replace('해설', '실습')
            to_file = to_file[:-6] + post_fix
        else:
            to_file = from_file[:-6] + post_fix
    
    # 파일이름 저장 (메타데이터에 적용)
    f_name = to_file.split('.')[0]
    f_name = f_name.split('/')[-1]

    if folder_path is not None:
        # 폴더 경로 없으면 생성
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # 폴더 경로를 포함한 파일 경로 생성
        to_file = os.path.join(folder_path, os.path.basename(to_file))

    # 메타데이터
    colab = y['metadata'].get('colab')
    if colab:
        y['metadata']['colab']['name'] = f_name


    with open(to_file, "w") as json_file:
        json.dump(y, json_file)
    print('생성완료')
    print(f'파일명: {to_file}')
    

# folder_path: 변환할 폴더 경로
# new_folder_name: 기본값은 /자동변환. 새로 생성할 폴더명
def convert_ipynb_folder(folder_path, new_folder_name='변환', post_fix='-변환.ipynb', remain_code=False):
    """
    folder_path: 변환할 폴더 경로
    new_folder_name: 기본값은 /자동변환. 새로 생성할 폴더명
    remain_code: default: False. True 적용시 # 코드입력 Cell의 코드를 살려둠
    
    (예시)
    convert_ipynb_folder(folder_path, new_folder_name='실습폴더', post_fix='.ipynb')
    
    변환 (post_fix 적용)
    convert_ipynb_folder(folder_path, post_fix='-자동변환.ipynb')
    """
    new_folder_path = os.path.join(folder_path, new_folder_name)
    
    if not os.path.isdir(new_folder_path):
        os.mkdir(new_folder_path)

    ipynb_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('ipynb')]

    for file in ipynb_list:
        convert_ipynb(file, folder_path=new_folder_path, post_fix=post_fix, remain_code=remain_code)


### Util 함수
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error


class ModelPlot():
    def __init__(self, error='mse', figsize=(15, 10)):
        self.my_predictions = {}

        self.figsize = figsize
        self.font_big = 15
        self.font_small = 12
        self.graph_width = 10
        self.round = 5
        self.error_name, self.error = self.set_error(error)

        self.colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
                       'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
                       'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
                       'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
                       'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
                       'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
                       ]

    def set_plot_options(self, figsize=(15, 10), font_big=15, font_small=12, graph_width=10, round=5):
        self.figsize = figsize
        self.font_big = font_big
        self.font_small = font_small
        self.graph_width = graph_width
        self.round = round

    def set_error(self, error='mse'):
        if error == 'mse':
            self.error = mean_squared_error
            self.error_name = 'mse'
            return error, self.error
        elif error == 'rmse':
            def rmse(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred))
            self.error = rmse
            self.error_name = 'rmse'
            return error, self.error

        elif error == 'rmsle':
            def rmsle(y_true, y_pred):
                return np.sqrt(mean_squared_log_error(y_true, y_pred))
            self.error = rmsle
            self.error_name = 'rmsle'
            return error, self.error
        elif error == 'mae':
            self.error = mean_absolute_error
            self.error_name = 'mae'
            return error, self.error
        else:
            self.error = mean_squared_error
            self.error_name = 'mse'
            return 'mse', mean_squared_error

    def plot_predictions(self, name_, actual, pred):
        df = pd.DataFrame({'prediction': pred, 'actual': actual})
        df = df.sort_values(by='actual').reset_index(drop=True)

        plt.figure(figsize=self.figsize)
        plt.scatter(df.index, df['prediction'], marker='x', color='r')
        plt.scatter(df.index, df['actual'], alpha=0.7, marker='o', color='black')
        plt.title(name_, fontsize=self.font_big)
        plt.legend(['prediction', 'actual'], fontsize=self.font_small)
        plt.show()

    def plot_error(self, name_, actual, pred):
        pred = np.asarray(pred).reshape(-1)
        actual = np.asarray(actual).reshape(-1)

        self.plot_predictions(name_, actual, pred)

        err = self.error(actual, pred)
        self.my_predictions[name_] = err

        y_value = sorted(self.my_predictions.items(), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(y_value, columns=['model', 'error'])

        display(df)

        min_ = max(df['error'].min() - 10, 0)
        max_ = df['error'].max()
        diff = (max_ - min_)
        max_ += diff * 0.25
        offset = diff * 0.05

        length = len(df) / 2

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(self.graph_width, length)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df['model'], fontsize=self.font_small)

        bars = ax.barh(np.arange(len(df)), df['error'], height=0.3)

        for i, v in enumerate(df['error']):
            idx = np.random.choice(len(self.colors))
            bars[i].set_color(self.colors[idx])
            ax.text(v + offset, i, str(round(v, self.round)), color='k', fontsize=self.font_small, fontweight='bold',
                    verticalalignment='center')

        ax.set_title(f'{self.error_name.upper()} Error', fontsize=self.font_big)
        ax.set_xlim(min_, max_)

        plt.show()

    def plot_all(self):
        y_value = sorted(self.my_predictions.items(), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(y_value, columns=['model', 'error'])

        display(df)

        min_ = max(df['error'].min() - 10, 0)
        max_ = df['error'].max()
        diff = (max_ - min_)
        max_ += diff * 0.25
        offset = diff * 0.05

        length = len(df) / 2

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(self.graph_width, length)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df['model'], fontsize=self.font_small)

        bars = ax.barh(np.arange(len(df)), df['error'], height=0.3)

        for i, v in enumerate(df['error']):
            idx = np.random.choice(len(self.colors))
            bars[i].set_color(self.colors[idx])
            ax.text(v + offset, i, str(round(v, self.round)), color='k', fontsize=self.font_small, fontweight='bold',
                    verticalalignment='center')

        ax.set_title(f'{self.error_name.upper()} Error', fontsize=self.font_big)
        ax.set_xlim(min_, max_)

        plt.show()

    def add_model(self, name_, actual, pred):
        err = self.error(actual, pred)
        self.my_predictions[name_] = err

    def remove_model(self, name_):
        if name_ in self.my_predictions:
            self.my_predictions.pop(name_)
        else:
            print('(에러) 지정한 키 값으로 등록된 모델이 없습니다.')

    def clear_error(self):
        self.my_predictions.clear()


plot = ModelPlot(error='mse')


def set_plot_error(error='mse'):
    global plot
    '''
    error: 'mse', 'rmse', 'rmsle', 'mae'
    '''
    plot.set_error(error)


def plot_error(name_, actual, prediction):
    global plot
    plot.plot_error(name_, actual, prediction)


def plot_all():
    global plot
    plot.plot_all()


def remove_error(name_):
    global plot
    plot.remove_model(name_)


def add_error(name_, actual, prediction):
    global plot
    plot.add_model(name_, actual, prediction)


def clear_error():
    global plot
    plot.clear_error()


def set_plot_options(figsize=(15, 10), font_big=15, font_small=12, graph_width=10, round=5):
    global plot
    plot.set_plot_options(figsize=figsize, font_big=font_big, font_small=font_small, graph_width=graph_width, round=round)