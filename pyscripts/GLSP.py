import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from loading import read_volt_grid
from sklearn.preprocessing import StandardScaler


class gl_model():
    def __init__(self, pred_str=0, target_sensor_count=False, apply_norm=True):
        self.pred_str = pred_str
        self.apply_norm = apply_norm
        self.scaler = StandardScaler()        
        self.target_sensor_count = target_sensor_count
    def init_selector(self,alpha=1.0, target_sensor_count=5, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.selector = gl_sensor_selector(alpha=alpha,target_sensor_count=5 ,fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)
    def init_predicor(self):
        self.predictor = BaggingRegressor(base_estimator=LinearRegression(fit_intercept=False), n_estimators=5, n_jobs=6)
        #self.predictor = LinearRegression(fit_intercept=False)
    def fit(self, data_train):
        data_selector = data_train[:,:105]
        x = data_selector[::2,:data_selector.shape[1]-self.pred_str].T
        y = data_selector[1::2,self.pred_str:].T
        if self.apply_norm:
            x = self.scaler.fit_transform(x)    
            y = self.scaler.fit_transform(y)
        parameters = {'alpha':np.arange(0.65, 0.75, 0.02)}
        # sensor selection
        self.init_selector(max_iter=10000,fit_intercept=False,positive=True, target_sensor_count=self.target_sensor_count)
        self.init_predicor()
        score_correlation = make_scorer(self.selector.loss_correlation, greater_is_better=False)
        score_sensor_count = make_scorer(self.selector.loss_sensor_count, greater_is_better=False)
        if self.target_sensor_count:
            self.cv = GridSearchCV(self.selector, parameters, cv=2, refit= 'count', scoring={'correlation':score_correlation, 'count':score_sensor_count}, n_jobs=6)
        else:
            self.cv = GridSearchCV(self.selector, parameters, cv=2, refit= 'correlation', scoring={'correlation':score_correlation, 'count':score_sensor_count}, n_jobs=1)
        self.cv.fit(X=y, y=x)
        self.validity = self.cv.best_estimator_.validity
        # data filtering
        if self.validity:
            self.sensor_map = self.cv.best_estimator_.predict(0)
            self.selected_sensors = np.zeros(shape=data_train.shape[0], dtype=bool)
            print(np.sum(self.sensor_map))
            self.selected_sensors[::2] = self.sensor_map
            self.selected_x = data_train[self.selected_sensors,:data_train.shape[1]-self.pred_str].T
            other_y = data_train[np.bitwise_not(self.selected_sensors), self.pred_str:].T
            self.predictor.fit(X=self.selected_x, y=other_y)
    def retrain_pred(self, data_train):
        self.selected_x = data_train[self.selected_sensors,:data_train.shape[1]-self.pred_str].T
        other_y = data_train[np.bitwise_not(self.selected_sensors), self.pred_str:].T
        self.predictor.fit(X=self.selected_x, y=other_y)
    def predict(self, x):
        X = x[:,self.selected_sensors]
        return self.predictor.predict(X)
    def evaluate(self, x, y):
        X = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(X)
        return [0, mean_squared_error(y, y_pred)]
class gl_sensor_selector(Lasso):
    def __init__(self, alpha=1.0, target_sensor_count=5,fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.target_sensor_count = target_sensor_count
        super().__init__(alpha=alpha,fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)
    def fit(self, true_y, true_x):
        # when call fit, sklearn always takes the 1st input as x and 2nd input as y.
        # when call score, skleran always takes the 1st input as y and 2nd input as prediction
        # but we want socre function take x instead of y.
        # this trick is to get around the score function limitation
        x = true_x
        y = true_y
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)
        super().fit(x,y)
        w = self.coef_
        if w.ndim == 2:
            self.importance = np.linalg.norm(w, axis=1)
        else:
            self.importance = w
        # cluster = KMeans(n_clusters=2)
        # cluster.fit(self.importance.reshape(-1,1))
        # found_cluster_count = np.unique(cluster.labels_)
        # # selecting bigger center
        # cluster_means = []
        # for label in [0,1]:
        #     clustered_nodes = cluster.labels_ == label
        #     cluster_means.append(np.mean(w[clustered_nodes]))
        # if found_cluster_count.size == 1:
        #     self.validity = False
        # else:
        #     self.validity = True
        # self.larger_center = np.argmax(cluster_means)
        # self.selected = cluster.labels_ == self.larger_center
        self.validity = True
        self.selected = np.zeros_like(self.importance, dtype=bool)
        sorted_w = np.argsort(self.importance)
        self.selected[sorted_w[-17:]] = True
        print(self.selected[:20])
        #print(np.linalg.norm(y.T - w.dot(x.T),1))
    def predict(self, x):
        if self.validity:
            return self.selected
        else:
            return False
    def loss_sensor_count(self, y_true, selected):
        if type(selected) != bool:
            return np.abs(np.count_nonzero(selected)-self.target_sensor_count)
        else:
            return 100000

    def loss_correlation(self, y_true, selected):
        if type(selected) != bool:
            X = pd.DataFrame(y_true[:, selected])
            corrmat = X.corr()
            mask = np.ones(corrmat.shape, dtype='bool')
            mask[np.triu_indices(len(corrmat))] = False
            z_trans = np.arctan(corrmat.values)
            z_mean  = np.mean(np.absolute(z_trans))
            return np.abs(np.tanh(z_mean))
        else:
            return 1
def loss_sensor_count(y_true, selected):
    return np.count_nonzero(selected)

def loss_correlation(y_true, selected):
    X = pd.DataFrame(y_true[:, selected])
    corrmat = X.corr()
    mask = np.ones(corrmat.shape, dtype='bool')
    mask[np.triu_indices(len(corrmat))] = False
    z_trans = np.arctan(corrmat.values)
    z_mean  = np.mean(np.absolute(z_trans))
    return np.abs(np.tanh(z_mean))



if __name__ == "__main__":
    fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    n = 100
    data = read_volt_grid(fname, n)
    models = []
    registered_count = []
    for pred_str in [1,2,3,4,5]:
        glsp = gl_model(pred_str=pred_str)
        glsp.fit(data)
        if glsp.validity:
            models.append(glsp)
            registered_count.append(pred_str)
        else:
            print(pred_str)
        import pickle
        pickle.dump(registered_count, open("gl.pred_str.registry1","wb"))
        pickle.dump(models, open("gl.pred_str.models1","wb"))
    # fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    # n = 200
    # data = read_volt_grid(fname, n)
    # models = []
    # registered_count = []
    # for target_sensor_count in [20,40,80,160,320]:
    #     glsp = gl_model(target_sensor_count=target_sensor_count, pred_str=10)
    #     glsp.fit(data)
    #     if glsp.validity:
    #         models.append(glsp)
    #         registered_count.append(target_sensor_count)
    #     import pickle
    #     pickle.dump(registered_count, open("gl.target_sensor_count.registry1","wb"))
    #     pickle.dump(models, open("gl.target_sensor_count.models1","wb"))

