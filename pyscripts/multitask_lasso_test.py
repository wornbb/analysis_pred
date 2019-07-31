from sklearn.linear_model import MultiTaskLasso
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from loading import read_volt_grid
from sklearn.preprocessing import StandardScaler
class glsp(MultiTaskLasso):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=1000, tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic'):
        super().__init__(alpha=alpha,fit_intercept=fit_intercept,
            normalize=normalize, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            random_state=random_state,
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
        cluster.fit(self.importance.reshape(-1,1))
        self.larger_center = np.argmax(cluster.cluster_centers_)
        self.selected = cluster.labels_ == self.larger_center
        #print(np.linalg.norm(y.T - w.dot(x.T),1))

    # def predict(self, x):
    #     return self.selected

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
    x_train = data_train[::2,:].T
    y_train = data_train[1::4,:].T
    start = 1
    end = 100
    jump = 20
    parameters = {'alpha':np.arange(start, end, jump)}

    ls = glsp(alpha=25,max_iter=10000,fit_intercept=False)
    #ls.fit(y_train, x_train)
    y_slice = y_train[:, 0]
    beta = x_train.T * (y_train - x_train)
    yp = ls.predict(x_train)
    #print(np.linalg.norm(yp-y_train,2))
    print(np.mean(np.mean(yp)))
    print(np.mean(np.mean(y_train)))
    print(np.mean(np.mean(yp-y_train)))
    score_correlation = make_scorer(loss_correlation)
    score_sensor_count = make_scorer(loss_sensor_count)
    clf = GridSearchCV(ls, parameters, cv=2, refit= 'correlation', scoring={'correlation':score_correlation, 'count':score_sensor_count})

    clf.fit(y_train, x_train)
    print(clf.cv_results_)