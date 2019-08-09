import matplotlib as plt
import pickle
import h5py
import numpy as np
from keras.utils import to_categorical
import pandas
from sklearn.metrics import confusion_matrix
from GLSP import *
class benchmark_factory():
    def __init__(self, model_fname, data_list, exp_name):
        self.model_fname = model_fname
        self.data_list = data_list
        self.models = self.load_benchmark_models(model_fname)
        self.benchmarks = self.load_benchmark_data(data_list)
        self.exp_name = exp_name
    def regression_mode_predict(self, model, x, y):
        regression = model.predict(x)
        prediction = np.bitwise_and(regression >= 1.04, regression <= 0.96)
    def benchmarking(self, mode):
        result_template = {"accs":[], "tp":0,"fp":0,"tn":0,"fn":0}
        if type(self.models) is not list:
            self.models = [self.models]
        self.evaluation = []
        for model in self.models:
            new_result = result_template
            for benchmark in self.benchmarks:

            self.evaluation.append(new_result)


    def generate_avg_acc_plt_data(self):
        fname = self.exp_name + ".avg_metric_curve.data"
        plt_data = {"model_order": self.models, "x": range(len(self.models))}
        all_y = []
        for model in self.models:
            y = []
            for benchmark in self.benchmarks:
                score = model.evaluate(benchmark[0], benchmark[1])
                y.append(score[1])
            all_y.append(np.mean(y ))
        plt_data.y = all_y
        pickle.dump([plt_data, self.model_fname], open(fname, 'wb'))
    def save_acc_tbl(self):
        fname = self.exp_name + ".acc_table.data"
        all_y = []
        for benchmark in self.benchmarks:
            y = []
            for model in self.models:
                score = model.evaluate(benchmark[0], benchmark[1])
                y.append(score[1])
            all_y.append(y)
        df = pandas.DataFrame(np.array(all_y), index=self.data_list, columns=self.model_fname)
        with open(fname, 'w') as latex:
            latex.write(df.to_latex())
    def generate_sensor_selection_data(self):
        fname = self.exp_name + ".selected_sensor_grid.data"
        all_sensors = []
        for model in self.models:
            sensors = model.selected_sensors
            dim = len(sensors)
            row = int(np.sqrt(dim))
            all_sensors.append(sensors.reshape((row, row)))
        pickle.dump([all_sensors, self.model_list], open(fname, 'wb'))
    def generate_confusion_matrix_data(self):
        fname = self.exp_name + ".confusion_matrix.data"
        all_matrix = []
        for model in self.models:
            matrix = np.zeros((2,2))
            for benchmark in self.benchmarks:
                y_pred = model.predict(benchmark[0])
                y_pred = np.argmax(y_pred, axis=-1)
                y_true = np.argmax(benchmark[1], axis=-1)
                matrix += confusion_matrix(y_true, y_pred)
            all_matrix.append(matrix)
        pickle.dump([all_matrix, self.model_list], open(fname, 'wb'))
    def load_benchmark_models(self, model_fname):
        #self.models = []
        #for fname in model_list:
        if model_fname.endswith(".h5"):
            print("h5 loading not implemented yet")
        else:
            saved_model = pickle.load(open(model_fname, 'rb'))
        self.models= saved_model
        return self.models
    def load_benchmark_data(self, data_list):
        data = []
        for fname in data_list:
            with h5py.File(fname, 'r') as f:
                x = f["data"][()]
                tag = f["tag"][()]
                tag = np.bitwise_not(tag < 1.5)
                #tag = to_categorical(tag)
            data.append([x, tag])
        return data

if __name__ == "__main__":
    data_list = ["balanced_gird_sensor.freqmine2c.h5", "balanced_gird_sensor.facesim2c.h5"]
    gp_models = "models_correct_score"
    gp_benchmark = benchmark_factory(gp_models, data_list,"gp")