from sklearn.linear_model import Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from loading import read_volt_grid
from sklearn.preprocessing import StandardScaler
class gl_model():
    def init_selector(self, alpha=1.0, fit_intercept=True, normalize=False,
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
    def fit(self, x, y):
        start = 10
        end = 100
        jump = 20
        parameters = {'alpha':np.arange(start, end, jump) / (2*y.shape[0])}
        # sensor selection
        self.init_selector(max_iter=10000,fit_intercept=False,positive=True)
        self.init_predicor()
        score_correlation = make_scorer(loss_correlation, greater_is_better=False)
        score_sensor_count = make_scorer(loss_sensor_count)
        self.cv = GridSearchCV(self.selector, parameters, cv=2, refit= 'correlation', scoring={'correlation':score_correlation, 'count':score_sensor_count})
        self.cv.fit(X=y, y=x)
        self.selected_sensors = self.cv.best_estimator_.predict(0)
        # data filtering
        self.selected_x = x[:,self.selected_sensors]
        self.predictor.fit(X=self.selected_x, y=y)
    def predict(self, x):
        x = x[:,self.selected_sensors]
        return self.predictor.predict(x)
    def evaluate(self, x, y):
        x = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(x)
        return [0, mean_squared_error(y, y_pred)]
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

def loss_sensor_count(y_true, selected):
    return np.count_nonzero(selected)

def loss_correlation(y_true, selected):
    X = pd.DataFrame(y_true[:, selected])
    corrmat = X.corr()
    mask = np.ones(corrmat.shape, dtype='bool')
    mask[np.triu_indices(len(corrmat))] = False
    z_trans = np.arctan(corrmat.values)
    z_mean  = np.mean(np.absolute(z_trans))
    return np.tanh(z_mean)


if __name__ == "__main__":
    fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    n = 40
    data = read_volt_grid(fname, n)
    [data_test, data_train] = np.split(data,2,axis=1)
    models = []
    for pred_str in range(8):
        pred_str += 1
        x_train = data_train[::2,:-pred_str].T
        y_train = data_train[1::2,pred_str:].T
        glsp = gl_model()
        glsp.fit(x_train, y_train)
        models.append(glsp)
    print('a')