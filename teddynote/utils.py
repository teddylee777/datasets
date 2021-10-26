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