from sklearn.linear_model import Lasso, LassoCV
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from loading import read_volt_grid
from sklearn.preprocessing import StandardScaler
class glsp(Lasso):
    def fit(self, x, y):
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)
        super().fit(x,y)
        w = self.coef_
        if w.ndim == 2:
            importance = np.linalg.norm(w, axis=0)
        else:
            importance = w
        cluster = KMeans(n_clusters=2)
        cluster.fit(importance.reshape(-1,1))
        self.larger_center = np.argmax(cluster.cluster_centers_)
        self.selected = cluster.labels_ == self.larger_center
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
    data = read_volt_grid(fname,40)
    [x_test, x_train] = np.split(data,2,axis=1)
    parameters = {'alpha':np.arange(0.01,0.1,1)}

    ls = glsp(warm_start=True,positive=True,selection='random',tol=1e-4*2)
    score_correlation = make_scorer(loss_correlation)
    score_sensor_count = make_scorer(loss_sensor_count)
    clf = GridSearchCV(ls, parameters, cv=2, refit= 'correlation', scoring={'correlation':score_correlation, 'count':score_sensor_count})
    clf.fit(x_train.T[::2,::2],x_train.T[1::2,1::2])
    print(sorted(clf.cv_results_.keys()))