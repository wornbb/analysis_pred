import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from loading import read_volt_grid


class gl_model():
    def __init__(self, pred_str=0, target_sensor_count=False):
        self.pred_str = pred_str
        if target_sensor_count:
            self.target_sensor_count = target_sensor_count
    def init_selector(self,alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.selector = gl_sensor_selector(alpha=alpha,fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)
    def init_predicor(self):
        self.predictor = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=5, n_jobs=6)
    def fit(self, data_train):
        x = data_train[::2,:-self.pred_str].T
        y = data_train[1::2,self.pred_str:].T
        parameters = {'alpha':np.arange(0.05, 1, 0.05)}
        # sensor selection
        self.init_selector(max_iter=10000,fit_intercept=False,positive=True)
        self.init_predicor()
        score_correlation = make_scorer(self.loss_correlation, greater_is_better=False)
        score_sensor_count = make_scorer(self.loss_sensor_count, greater_is_better=False)
        self.cv = GridSearchCV(self.selector, parameters, cv=2, refit= 'correlation', scoring={'correlation':score_correlation, 'count':score_sensor_count}, n_jobs=6)
        self.cv.fit(X=y, y=x)
        self.sensor_map = self.cv.best_estimator_.predict(0)
        self.selected_sensors = np.zeros(shape=data_train.shape[0], dtype=bool)
        self.selected_sensors[::2] = self.sensor_map
        # data filtering
        self.selected_x = data_train[self.selected_sensors,:-self.pred_str].T
        other_y = data_train[np.bitwise_not(self.selected_sensors), self.pred_str:].T
        self.predictor.fit(X=self.selected_x, y=other_y)
    def predict(self, x):
        X = x[:,self.selected_sensors]
        return self.predictor.predict(X)
    def evaluate(self, x, y):
        X = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(X)
        return [0, mean_squared_error(y, y_pred)]
    def loss_sensor_count(self, y_true, selected):
        return np.abs(np.count_nonzero(selected)-self.target_sensor_count)

    def loss_correlation(self, y_true, selected):
        X = pd.DataFrame(y_true[:, selected])
        corrmat = X.corr()
        mask = np.ones(corrmat.shape, dtype='bool')
        mask[np.triu_indices(len(corrmat))] = False
        z_trans = np.arctan(corrmat.values)
        z_mean  = np.mean(np.absolute(z_trans))
        return np.abs(np.tanh(z_mean))
class gl_sensor_selector(Lasso):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
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
            self.importance = np.linalg.norm(w, axis=0)
        else:
            self.importance = w
        cluster = KMeans(n_clusters=2)
        print(self.alpha)
        cluster.fit(self.importance.reshape(-1,1))
        self.larger_center = np.argmax(cluster.cluster_centers_)
        self.selected = cluster.labels_ == self.larger_center
        #print(np.linalg.norm(y.T - w.dot(x.T),1))

    def predict(self, x):
        return self.selected




if __name__ == "__main__":
    #fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    n = 1000
    data = read_volt_grid(fname, n)
    # models = []
    # for pred_str in [5,10,20,40,80]:
    #     glsp = gl_model(pred_str=pred_str)
    #     glsp.fit(data)
    #     models.append(glsp)
    # import pickle
    # pickle.dump(models, open("ee.pred_str.models","wb"))
    fname = "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
    models = []
    for target_sensor_count in [20,40,80,160,320,640]:
        glsp = gl_model(target_sensor_count=target_sensor_count)
        glsp.fit(data)
        models.append(glsp)
    import pickle
    pickle.dump(models, open("ee.target_sensor_count.models","wb"))

