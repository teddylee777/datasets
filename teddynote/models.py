import optuna
from optuna.exceptions import ExperimentalWarning

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import warnings
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, log_loss

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")


class OptunaParam():
    def __init__(self, name, low=None, high=None, fixed_value=None, categorical_value=None, param_type='int'):
        self.name = name
        self.low = low
        self.high = high
        self.param_type = param_type
        self.fixed_value = fixed_value
        self.categorical_value = categorical_value

    def create_param(self, trial):
        if self.param_type == 'int':
            return trial.suggest_int(self.name, self.low, self.high)
        elif self.param_type == 'loguniform':
            return trial.suggest_loguniform(self.name, self.low, self.high)
        elif self.param_type == 'uniform':
            return trial.suggest_uniform(self.name, self.low, self.high)
        elif self.param_type == 'categorical':
            return trial.suggest_categorical(self.name, self.categorical_value)
        elif self.param_type == 'fixed':
            return self.fixed_value

    def __str__(self):
        if self.param_type == 'int':
            return f'name: {self.name}, low: {self.low}, high: {self.high}, type: {self.param_type}'
        elif self.param_type == 'loguniform':
            return f'name: {self.name}, low: {self.low}, high: {self.high}, type: {self.param_type}'
        elif self.param_type == 'uniform':
            return f'name: {self.name}, low: {self.low}, high: {self.high}, type: {self.param_type}'
        elif self.param_type == 'categorical':
            return f'name: {self.name}, categorical_value: {self.categorical_value} type: {self.param_type}'
        elif self.param_type == 'fixed':
            return f'name: {self.name}, fixed_value: {self.fixed_value}, type: {self.param_type}'


class OptunaParamGrid():
    def __init__(self):
        self.params = {'verbose': OptunaParam('verbose', fixed_value=-1, param_type='fixed'),
                       'lambda_l1': OptunaParam('lambda_l1', low=1e-8, high=5, param_type='loguniform'),
                       'lambda_l2': OptunaParam('lambda_l2', low=1e-8, high=5, param_type='loguniform'),
                       'path_smooth': OptunaParam('path_smooth', low=1e-8, high=1e-3, param_type='loguniform'),
                       'learning_rate': OptunaParam('learning_rate', low=1e-5, high=1e-1, param_type='loguniform'),
                       'feature_fraction': OptunaParam('feature_fraction', low=0.5, high=0.9, param_type='uniform'),
                       'bagging_fraction': OptunaParam('bagging_fraction', low=0.5, high=0.9, param_type='uniform'),
                       'num_leaves': OptunaParam('num_leaves', low=15, high=90, param_type='int'),
                       'min_data_in_leaf': OptunaParam('min_data_in_leaf', low=10, high=100, param_type='int'),
                       'max_bin': OptunaParam('max_bin', low=100, high=255, param_type='int'),
                       'n_estimators': OptunaParam('n_estimators', low=100, high=3000, param_type='int'),
                       'bagging_freq': OptunaParam('bagging_freq', low=0, high=15, param_type='int'),
                       'min_child_weight': OptunaParam('min_child_weight', low=1, high=20, param_type='int'),
                       }

    def set_param(self, optuna_param: OptunaParam):
        self.params[optuna_param.name] = optuna_param

    def get_param(self, name):
        return self.params[name]

    def remove_param(self, name):
        self.params.pop(name)

    def set_paramgrid(self, param_grid):
        self.params = param_grid

    def print_params(self):
        for k, v in self.params.items():
            print(v)

    def create_paramgrid(self, trial):
        p = {}
        for k, v in self.params.items():
            p[k] = v.create_param(trial)
        return p


class BaseOptuna():

    def __init__(self):
        self.param_grid = OptunaParamGrid()

    def set_param(self, optuna_param: OptunaParam):
        self.param_grid.set_param(optuna_param)

    def get_param(self, name):
        return self.param_grid[name]

    def print_params(self):
        self.param_grid.print_params()

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, **dataset):
        raise NotImplementedError("fit_model 메소드를 구현하여야 합니다")

    def objective_func(self, trial, eval_metric=None, cat_features=None, cv=5, seed=None, n_rounds=3000,
                       **datasets):
        kfold = KFold(n_splits=cv, shuffle=True, random_state=seed)

        X = datasets['x']
        Y = datasets['y']

        errors = []

        for train_idx, test_idx in kfold.split(X):
            x_train, x_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = Y[train_idx], Y[test_idx]

            err = self.fit_model(trial,
                                 eval_metric=eval_metric,
                                 cat_features=cat_features,
                                 seed=seed,
                                 n_rounds=n_rounds,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_test=x_test,
                                 y_test=y_test)

            errors.append(err)

        return np.mean(errors)

    def optimize(self, x, y, test_data=None, cat_features=None, eval_metric='f1', cv=5, seed=None, n_rounds=3000, n_trials=100):
        if eval_metric in ['f1', 'accuracy', 'precision', 'recall']:
            direction = 'maximize'
        else:
            direction = 'minimize'

        self.study = optuna.create_study(direction=direction)
        dataset = {
            'x': x,
            'y': y
        }

        self.study.optimize(
            lambda trial: self.objective_func(trial, eval_metric, cat_features, cv=cv, seed=seed, n_rounds=n_rounds,
                                             **dataset), n_trials=n_trials)

        if test_data is not None:
            predictions = self.predict(dataset['x'], dataset['y'], test_data)
            return self.study.best_trial.params, predictions
        else:
            return self.study.best_trial.params, None

    def predict(self, x, y, test_data):
        raise NotImplementedError("predict 메소드를 구현하여야 합니다")

    def visualize(self):
        # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
        display(optuna.visualization.plot_param_importances(self.study))

        # 하이퍼파라미터 최적화 과정을 확인
        display(optuna.visualization.plot_optimization_history(self.study))

        return self.study.trials_dataframe().sort_values(by='value', ascending=False)

    def get_best_params(self):
        return self.study.best_trial.params


################ LGBM ################


class LGBMClassifierOptuna(BaseOptuna):

    def get_feval(self, num_classes, metrics, average='weighted'):
        def feval(y_pred, dataset):
            y_true = dataset.get_label()

            if num_classes < 3:
                y_hat = np.where(y_pred < 0.5, 0, 1)
                avg = 'binary'
            else:
                y_pred = y_pred.reshape(-1, num_classes)
                y_hat = np.argmax(y_pred, axis=1)
                avg = average

            if metrics == 'logloss':
                score = log_loss(y_true, y_pred)
                return 'score', score, False
            else:
                if metrics == 'f1':
                    score = f1_score(y_true, y_hat, average=avg)
                elif metrics == 'accuracy':
                    score = accuracy_score(y_true, y_hat)
                elif metrics == 'precision':
                    score = precision_score(y_true, y_hat, average=avg)
                elif metrics == 'recall':
                    score = recall_score(y_true, y_hat, average=avg)
                else:
                    score = accuracy_score(y_true, y_hat)
                return 'score', score, True

        return feval

    def evaluate(self, y_true, y_pred, num_classes, metrics, average='weighted'):
        if num_classes < 3:
            y_hat = np.where(y_pred < 0.5, 0, 1)
            avg = 'binary'
        else:
            y_pred = y_pred.reshape(-1, num_classes)
            y_hat = np.argmax(y_pred, axis=1)
            avg = average

        if metrics == 'logloss':
            score = log_loss(y_true, y_pred)
        else:
            if metrics == 'f1':
                score = f1_score(y_true, y_hat, average=avg)
            elif metrics == 'accuracy':
                score = accuracy_score(y_true, y_hat)
            elif metrics == 'precision':
                score = precision_score(y_true, y_hat, average=avg)
            elif metrics == 'recall':
                score = recall_score(y_true, y_hat, average=avg)
            else:
                score = accuracy_score(y_true, y_hat)

        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        dtrain = lgb.Dataset(dataset['x_train'],
                             label=dataset['y_train'],
                             categorical_feature=cat_features
                             )
        dtest = lgb.Dataset(dataset['x_test'],
                            label=dataset['y_test'],
                            categorical_feature=cat_features
                            )

        num_classes = np.unique(dataset['y_train']).shape[0]

        if num_classes < 3:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            params['num_class'] = 1
        else:
            params['objective'] = 'multiclass'  # Multi-class
            params['metric'] = 'multi_logloss'  # metric for multi-class
            params['num_class'] = num_classes

        params['random_state'] = seed

        feval = self.get_feval(num_classes, eval_metric)

        gbm = lgb.train(params, dtrain,
                        early_stopping_rounds=30,
                        num_boost_round=n_rounds,
                        feval=feval,
                        valid_sets=[dtrain, dtest],
                        verbose_eval=-1,
                        )

        preds = gbm.predict(dataset['x_test'])
        err = self.evaluate(dataset['y_test'], preds, num_classes, eval_metric, average='weighted')
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
        model = lgb.LGBMClassifier(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds



class LGBMRegressorOptuna(BaseOptuna):

    def get_feval(self, metrics):
        def feval(y_pred, dataset):
            y_true = dataset.get_label()
            if metrics == 'mse':
                score = mean_squared_error(y_true, y_pred)
            elif metrics == 'mae':
                score = mean_absolute_error(y_true, y_pred)
            elif metrics == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metrics == 'rmsle':
                score = np.sqrt(mean_squared_log_error(y_true, y_pred))
            else:
                score = mean_squared_error(y_true, y_pred)
            return 'score', score, False

        return feval

    def evaluate(self, y_true, y_pred, metrics):
        if metrics == 'mse':
            score = mean_squared_error(y_true, y_pred)
        elif metrics == 'mae':
            score = mean_absolute_error(y_true, y_pred)
        elif metrics == 'rmse':
            score = np.sqrt(mean_squared_error(y_true, y_pred))
        elif metrics == 'rmsle':
            score = np.sqrt(mean_squared_log_error(y_true, y_pred))
        else:
            score = mean_squared_error(y_true, y_pred)
        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        dtrain = lgb.Dataset(dataset['x_train'],
                             label=dataset['y_train'],
                             categorical_feature=cat_features
                             )
        dtest = lgb.Dataset(dataset['x_test'],
                            label=dataset['y_test'],
                            categorical_feature=cat_features
                            )

        if eval_metric == 'mae':
            params['objective'] = 'regression_l1'
            params['metric'] = 'l1'
        elif eval_metric == 'mse':
            params['objective'] = 'regression'
            params['metric'] = 'l2'
        elif eval_metric == 'rmse':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:
            params['objective'] = 'regression'
            params['metric'] = 'l2'

        params['random_state'] = seed

        feval = self.get_feval(eval_metric)
        gbm = lgb.train(params, dtrain,
                        early_stopping_rounds=30,
                        num_boost_round=n_rounds,
                        feval=feval,
                        valid_sets=[dtrain, dtest],
                        verbose_eval=-1,
                        )

        preds = gbm.predict(dataset['x_test'])
        err = self.evaluate(dataset['y_test'], preds, eval_metric)
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        model = lgb.LGBMRegressor(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds

################ XGBoost ################

class XGBClassifierOptuna(BaseOptuna):

    def __init__(self, use_gpu=False):
        super().__init__()
        params = {'nthread': OptunaParam('nthread', fixed_value=-1, param_type='fixed'),
                  'lambda': OptunaParam('lambda', low=1e-5, high=5, param_type='loguniform'),
                  'alpha': OptunaParam('alpha', low=1e-5, high=5, param_type='loguniform'),
                  'colsample_bytree': OptunaParam('colsample_bytree', low=0.5, high=0.9, param_type='uniform'),
                  'subsample': OptunaParam('subsample', low=0.5, high=0.9, param_type='uniform'),
                  'learning_rate': OptunaParam('learning_rate', low=1e-5, high=1e-1, param_type='loguniform'),
                  'n_estimators': OptunaParam('n_estimators', low=100, high=5000, param_type='int'),
                  'max_depth': OptunaParam('max_depth', low=6, high=30, param_type='int'),
                  'min_child_weight': OptunaParam('min_child_weight', low=1, high=300, param_type='int'),
                  'verbosity': OptunaParam('verbosity', fixed_value=0, param_type='fixed'),
                  }
        if use_gpu:
            params['tree_method'] = OptunaParam('tree_method', fixed_value='gpu_hist', param_type='fixed')
        self.param_grid.set_paramgrid(params)

    def evaluate(self, y_true, y_pred, num_classes, metrics, average='weighted'):
        if num_classes < 3:
            y_hat = np.where(y_pred < 0.5, 0, 1)
            avg = 'binary'
        else:
            y_hat = y_pred
            avg = average

        if metrics == 'logloss':
            score = log_loss(y_true, y_pred)
        else:
            if metrics == 'f1':
                score = f1_score(y_true, y_hat, average=avg)
            elif metrics == 'accuracy':
                score = accuracy_score(y_true, y_hat)
            elif metrics == 'precision':
                score = precision_score(y_true, y_hat, average=avg)
            elif metrics == 'recall':
                score = recall_score(y_true, y_hat, average=avg)
            else:
                score = accuracy_score(y_true, y_hat)

        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        dtrain = xgb.DMatrix(dataset['x_train'],
                             label=dataset['y_train'])

        dtest = xgb.DMatrix(dataset['x_test'],
                            label=dataset['y_test'])

        num_classes = np.unique(dataset['y_train']).shape[0]

        if num_classes < 3:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        else:
            params['objective'] = 'multi:softmax'  # Multi-class
            params['eval_metric'] = 'mlogloss'  # Metric for Multi-class
            params['num_class'] = num_classes

        params['random_state'] = seed

        gbm = xgb.train(params, dtrain=dtrain,
                        evals=[(dtrain, 'train'), (dtest, 'eval')],
                        early_stopping_rounds=50,
                        num_boost_round=n_rounds,
                        verbose_eval=0
                        )

        preds = gbm.predict(dtest)
        err = self.evaluate(dataset['y_test'], preds, num_classes, eval_metric, average='weighted')
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
        model = xgb.XGBClassifier(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds


class XGBRegressorOptuna(BaseOptuna):

    def __init__(self, use_gpu=False):
        super().__init__()
        params = {'nthread': OptunaParam('nthread', fixed_value=-1, param_type='fixed'),
                  'lambda': OptunaParam('lambda', low=1e-5, high=5, param_type='loguniform'),
                  'alpha': OptunaParam('alpha', low=1e-5, high=5, param_type='loguniform'),
                  'colsample_bytree': OptunaParam('colsample_bytree', low=0.5, high=0.9, param_type='uniform'),
                  'subsample': OptunaParam('subsample', low=0.5, high=0.9, param_type='uniform'),
                  'learning_rate': OptunaParam('learning_rate', low=1e-5, high=1e-1, param_type='loguniform'),
                  'n_estimators': OptunaParam('n_estimators', low=100, high=5000, param_type='int'),
                  'max_depth': OptunaParam('max_depth', low=6, high=30, param_type='int'),
                  'min_child_weight': OptunaParam('min_child_weight', low=1, high=300, param_type='int'),
                  'verbosity': OptunaParam('verbosity', fixed_value=0, param_type='fixed'),
                  }
        if use_gpu:
            params['tree_method'] = OptunaParam('tree_method', fixed_value='gpu_hist', param_type='fixed')
        self.param_grid.set_paramgrid(params)

    def evaluate(self, y_true, y_pred, metrics):
        if metrics == 'mse':
            score = mean_squared_error(y_true, y_pred)
        elif metrics == 'mae':
            score = mean_absolute_error(y_true, y_pred)
        elif metrics == 'rmse':
            score = np.sqrt(mean_squared_error(y_true, y_pred))
        elif metrics == 'rmsle':
            score = np.sqrt(mean_squared_log_error(y_true, y_pred))
        else:
            score = mean_squared_error(y_true, y_pred)
        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        dtrain = xgb.DMatrix(dataset['x_train'],
                             label=dataset['y_train'])

        dtest = xgb.DMatrix(dataset['x_test'],
                            label=dataset['y_test'])

        if eval_metric == 'mae':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'mae'
        elif eval_metric == 'mse':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        elif eval_metric == 'rmse':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        elif eval_metric == 'rmsle':
            params['objective'] = 'reg:squaredlogerror'
            params['eval_metric'] = 'rmsle'
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'

        params['random_state'] = seed

        gbm = xgb.train(params, dtrain=dtrain,
                        evals=[(dtrain, 'train'), (dtest, 'eval')],
                        early_stopping_rounds=50,
                        num_boost_round=n_rounds,
                        verbose_eval=0
                        )

        preds = gbm.predict(dtest)
        err = self.evaluate(dataset['y_test'], preds, eval_metric)
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        model = xgb.XGBRegressor(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds


################ CatBoost ################

class CatBoostClassifierOptuna(BaseOptuna):

    def __init__(self, use_gpu=False):
        super().__init__()
        params = {'bootstrap_type': OptunaParam('bootstrap_type', categorical_value=['Bayesian', 'Bernoulli', 'MVS'], param_type='categorical'),
                  'boosting_type': OptunaParam('boosting_type', categorical_value=['Ordered', 'Plain'], param_type='categorical'),
                  'od_type': OptunaParam('od_type', categorical_value=['IncToDec', 'Iter'], param_type='categorical'),
                  'colsample_bylevel': OptunaParam('colsample_bylevel', low=0.01, high=0.1, param_type='uniform'),
                  'l2_leaf_reg': OptunaParam('l2_leaf_reg', low=1.0, high=5.5, param_type='uniform'),
                  'learning_rate': OptunaParam('learning_rate', low=1e-2, high=5e-1, param_type='loguniform'),
                  'iterations': OptunaParam('iterations', low=100, high=2000, param_type='int'),
                  'min_child_samples': OptunaParam('min_child_samples', low=1, high=32, param_type='int'),
                  'depth': OptunaParam('depth', low=2, high=12, param_type='int'),
                  'logging_level': OptunaParam('logging_level', fixed_value='Silent', param_type='fixed'),
                  }

        if use_gpu:
            params['task_type'] = OptunaParam('task_type', fixed_value='GPU', param_type='fixed')

        self.param_grid.set_paramgrid(params)

    def evaluate(self, y_true, y_pred, num_classes, metrics, average='weighted'):
        if num_classes < 3:
            y_hat = np.where(y_pred < 0.5, 0, 1)
            avg = 'binary'
        else:
            y_hat = y_pred
            avg = average

        if metrics == 'logloss':
            score = log_loss(y_true, y_pred)
        else:
            if metrics == 'f1':
                score = f1_score(y_true, y_hat, average=avg)
            elif metrics == 'accuracy':
                score = accuracy_score(y_true, y_hat)
            elif metrics == 'precision':
                score = precision_score(y_true, y_hat, average=avg)
            elif metrics == 'recall':
                score = recall_score(y_true, y_hat, average=avg)
            else:
                score = accuracy_score(y_true, y_hat)
        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.1, 50)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_uniform("subsample", 0.5, 0.9)

        num_classes = np.unique(dataset['y_train']).shape[0]

        if num_classes < 3:
            params['objective'] = 'Logloss'
        else:
            params['objective'] = 'MultiClass'  # Multi-class

        params['random_seed'] = seed
        params['cat_features'] = cat_features

        gbm = cb.CatBoostClassifier(**params)
        gbm.fit(dataset['x_train'], dataset['y_train'],
                eval_set=[(dataset['x_test'], dataset['y_test'])], verbose=0, early_stopping_rounds=30)

        preds = gbm.predict(dataset['x_test'])
        err = self.evaluate(dataset['y_test'], preds, num_classes, eval_metric, average='weighted')
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
        model = cb.CatBoostClassifier(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds


class CatBoostRegressorOptuna(BaseOptuna):

    def __init__(self, use_gpu=False):
        super().__init__()
        params = {'bootstrap_type': OptunaParam('bootstrap_type', categorical_value=['Bayesian', 'Bernoulli', 'MVS'], param_type='categorical'),
                  'boosting_type': OptunaParam('boosting_type', categorical_value=['Ordered', 'Plain'], param_type='categorical'),
                  'od_type': OptunaParam('od_type', categorical_value=['IncToDec', 'Iter'], param_type='categorical'),
                  'colsample_bylevel': OptunaParam('colsample_bylevel', low=0.01, high=0.1, param_type='uniform'),
                  'l2_leaf_reg': OptunaParam('l2_leaf_reg', low=1.0, high=5.5, param_type='uniform'),
                  'learning_rate': OptunaParam('learning_rate', low=1e-2, high=5e-1, param_type='loguniform'),
                  'iterations': OptunaParam('iterations', low=100, high=2000, param_type='int'),
                  'min_child_samples': OptunaParam('min_child_samples', low=1, high=32, param_type='int'),
                  'depth': OptunaParam('depth', low=2, high=12, param_type='int'),
                  'logging_level': OptunaParam('logging_level', fixed_value='Silent', param_type='fixed'),
                  }

        if use_gpu:
            params['task_type'] = OptunaParam('task_type', fixed_value='GPU', param_type='fixed')

        self.param_grid.set_paramgrid(params)

    def evaluate(self, y_true, y_pred, metrics):
        if metrics == 'mse':
            score = mean_squared_error(y_true, y_pred)
        elif metrics == 'mae':
            score = mean_absolute_error(y_true, y_pred)
        elif metrics == 'rmse':
            score = np.sqrt(mean_squared_error(y_true, y_pred))
        elif metrics == 'rmsle':
            score = np.sqrt(mean_squared_log_error(y_true, y_pred))
        else:
            score = mean_squared_error(y_true, y_pred)
        return score

    def fit_model(self, trial, eval_metric=None, cat_features=None, seed=None, n_rounds=3000, **dataset):
        params = self.param_grid.create_paramgrid(trial)

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.1, 50)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_uniform("subsample", 0.5, 0.9)

        if eval_metric == 'mae':
            params['objective'] = 'MAE'
        elif eval_metric == 'mse':
            params['objective'] = 'RMSE'
        elif eval_metric == 'rmse':
            params['objective'] = 'RMSE'
        elif eval_metric == 'rmsle':
            params['objective'] = 'msle'
        elif eval_metric == 'msle':
            params['objective'] = 'msle'
        else:
            params['objective'] = 'RMSE'

        params['random_seed'] = seed
        params['cat_features'] = cat_features

        gbm = cb.CatBoostRegressor(**params)
        gbm.fit(dataset['x_train'], dataset['y_train'],
                eval_set=[(dataset['x_test'], dataset['y_test'])], verbose=0, early_stopping_rounds=30)

        preds = gbm.predict(dataset['x_test'])
        err = self.evaluate(dataset['y_test'], preds, eval_metric)
        return err

    def predict(self, x, y, test_data):
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        model = cb.CatBoostRegressor(**self.get_best_params())
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=0, early_stopping_rounds=30)
        preds = model.predict(test_data)
        return preds

