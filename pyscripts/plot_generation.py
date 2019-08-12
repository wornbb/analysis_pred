import matplotlib as plt
import pickle
import h5py
import numpy as np
from keras.utils import to_categorical
import pandas
from sklearn.metrics import confusion_matrix
from GLSP import *
from loading import *
class benchmark_factory():
    def __init__(self, model_fname, data_list, exp_name, mode):
        self.model_fname = model_fname
        self.data_list = data_list
        self.models = self.load_benchmark_models(model_fname)
        self.loaded_benchmark = 0
        self.exp_name = exp_name
        self.mode = mode
        if mode == "regression":
            self.predictor = self.regression_mode_predict
        elif mode == "neural":
            self.predictor = self.neural_mode_predict
        # default parameters
        self.lines_to_read = 10000
    def blank_result(self):
        result = {"accs":0, "tp":0,"fp":0,"tn":0,"fn":0}
        if self.mode == "regression":
            result["regression_hit"] = 0
            result["regression_total"] = 0
        return result
    def regression_mode_predict(self, model, x):
        regression = model.predict(x[::2,:])
        violation = np.bitwise_and(regression >= 1.04, regression <= 0.96)
        if violation.any():
            prediction = 1
        else:
            prediction = 0
        return [prediction, regression]
    def neural_mode_predict(self):
        a = 1
    def evaluator(self, model, x, y):
        sample_size = x.shape[0]
        result = self.blank_result()
        for sample in range(sample_size):
            from_predictor = self.predictor(model, x[sample:sample+1,::2])
            result = self.test_prediction(from_predictor, x, y, sample, result)
        result = self.finalize_result(result)
        return result
    def finalize_result(self, result):
        result["acc"] = (result["tp"] + result["tn"]) / (result["fp"] + result["fn"])
        if self.mode == "regression":
            result["regression_acc"] = result["regression_acc"] / result["regression_total"]
        return result
    def test_prediction(self, from_predictor, x, y, sample, result):
        if self.mode == "regression":
            result['regression_total'] += 1
            prediction = from_predictor[0]
            regression = from_predictor[1]
            # dirty fixing
            pred_str = self.loaded_model
            # regression benchmarking
            target = x[sample + pred_str,1::2]
            error = np.absolute(regression - target)
            diff = error / regression
            max_diff = np.amax(diff)
            if max_diff <= 1/10**4:
                result["regression_acc"] += 1
        else:
            prediction = from_predictor
            pred_str = 5
        # register regression matrix
        key = ""
        if prediction == y[sample + pred_str]:
            key += "t"
        else:
            key += "f"
        if prediction == 1:
            key += "p"
        else:
            key += "n"
        result[key] += 1
        return result
    def benchmarking(self):
        if type(self.models) is not list:
            self.models = [self.models]
        self.all_evaluations = []
        self.evaluation = dict.fromkeys(self.data_list)
        self.loaded_model = 0
        for model in self.models:
            for dataset in self.data_list:
                benchmark = self.load_benchmark_data(dataset)
                result = self.evaluator(model, benchmark[0], benchmark[1])
                self.evaluation[dataset] = result
            self.all_evaluations.append(self.evaluation)
            self.loaded_model += 1
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
    def load_benchmark_data(self, fname):
        if fname.endswith(".h5"):
            with h5py.File(fname, 'r') as f:
                x = f["x"][()]
                tag = f["y"][()]
                tag = tag < 1.5
        elif fname.endswith(".gridIR"):
            loader = regression_training_data_factory([fname], lines_to_read=self.lines_to_read)
            [x, tag] = loader.generate()
            tag = np.array(tag)
            x = x.T
        else:
            print("undefined file type to load")
        return [x, tag.astype('int')]

if __name__ == "__main__":
    #data_list = [r"VoltNet_2c.h5"]
    data_list = [r"F:\\Yaswan2c\\Yaswan2c.gridIR"]
    gp_models = "models_correct_score"
    gp_benchmark = benchmark_factory(gp_models, data_list,exp_name="gp",mode="regression")
    gp_benchmark.benchmarking()