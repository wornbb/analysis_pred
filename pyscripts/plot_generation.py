import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import tikzplotlib
import pickle
import h5py
import numpy as np
from keras.utils import to_categorical
import pandas
from sklearn.metrics import confusion_matrix
from GLSP import *
from eagle import *
from loading import *
from pathlib import Path
import cv2
import os
from confusion_matrix_pretty_print import *
from voltNet import *
from sklearn.metrics import r2_score
from os.path import basename
import copy
class benchmark_factory():
    def __init__(self, model_flist, data_list, exp_name, mode, flp, pred_str_list, lines_to_read=1000, lines_to_jump=0):
        self.model_fname = model_flist
        self.data_list = data_list
        self.models = self.load_benchmark_models(model_flist)
        self.loaded_benchmark = 0
        self.exp_name = exp_name
        self.index_5 = 0
        self.mode = mode
        if mode == "regression":
            self.predictor = self.regression_mode_predict
        elif mode == "classification":
            self.predictor = self.neural_mode_predict
        self.flp = flp
        self.pred_str_list = pred_str_list
        # default parameters
        self.lines_to_read = lines_to_read
        self.lines_to_jump = lines_to_jump
        # directory magic
        self.save_prefix = self.exp_name + "." + self.mode 
        self.latex_fig = Path(r"./tex_f")
    def blank_result(self):
        """result template
        
        Returns:
            dict -- variable for holding the benchmark result
                    result = {"acc":0, "tp":0,"fp":0,"tn":0,"fn":0, ["regression_hit", "regression"]}
                    Before calling "self.finalize_result":
                        "acc": is a blank placeholder
                        "regression_acc": is a temporary buffer for holding the correct regression count
        """
        result = {"acc":0, "tp":0,"fp":0,"tn":0,"fn":0}
        if self.mode == "regression":
            result["regression_acc"] = 0
            result["regression_total"] = 0
            result["Mean_Squared_Error"] = 0
            result["Standard_Deviation_Error"] = 0
            result["R^2"] = 0
        return copy.deepcopy(result)
    def regression_mode_predict(self, model, x):
        # get the output from given model. the model should take care of the node selection
        # we benchmark a regression model should have 2 parts. 1. classificaiton 2. regression
        regression = model.predict(x)
        violation_pred = np.bitwise_or(regression >= 1.04, regression <= 0.96)
        violation_sample = np.bitwise_or(x[:,model.selected_sensors] >= 1.04, x[:,model.selected_sensors] <= 0.96)
        if violation_pred.any():
            prediction = 1
        # elif violation_sample.any():
        #     prediction = 1
        else:
            prediction = 0
        return [prediction, regression]
    def neural_mode_predict(self, model, x):
        # get the output from given model. the model should take care of the node selection
        # we benchmark a neural network model only on its classification performance
        prediction = model.predict(x)
        return prediction
    def evaluator(self, model, x, y):
        sample_size = x.shape[0]
        result = self.blank_result()
        #self.loaded_model is the prediction strength
        for sample in range(sample_size - self.pred_str):
            from_predictor = self.predictor(model, x[sample:sample+1,:])
            result = self.test_prediction(from_predictor, x, y, sample, result)
        result = self.finalize_result(result)
        return result
    def finalize_result(self, result):
        result["acc"] = (result["tp"] + result["tn"]) / (result["fp"] + result["fn"] + result["tp"] + result["tn"])
        if self.mode == "regression":
            result["regression_acc"] = result["regression_acc"] / result["regression_total"]
            result["Mean_Squared_Error"] /= (result["fp"] + result["fn"] + result["tp"] + result["tn"])
            result["R^2"] /= (result["fp"] + result["fn"] + result["tp"] + result["tn"])
            result["Standard_Deviation_Error"] /= (result["fp"] + result["fn"] + result["tp"] + result["tn"])
        return result
    def test_prediction(self, from_predictor, x, y, sample, result):
        if self.mode == "regression":
            result['regression_total'] += 1
            prediction = from_predictor[0]
            regression = from_predictor[1]
            # dirty fixing
            # regression benchmarking
            #target = x[sample + self.pred_str, np.bitwise_not(self.selected_sensors)]
            target = x[sample + self.pred_str, :]
            error = np.absolute(regression - target)
            diff = error / target
            max_diff = np.amax(diff)
            if max_diff <= 1/10**4:
                result["regression_acc"] += 1
            # stats
            std_y = np.std(target)
            std_p = np.std(regression.flatten())
            result["Standard_Deviation_Error"] += std_y - std_p
            result["Mean_Squared_Error"] += np.sum((target - regression.flatten())**2) / len(target)
            result["R^2"] += r2_score(target, regression.flatten())
        else:
            prediction = from_predictor
        # register regression matrix
        key = ""
        if prediction == y[sample + self.pred_str]:
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
        """
                    result = {"acc":0, "tp":0,"fp":0,"tn":0,"fn":0, ["regression_hit", "regression"]}
        """
        if type(self.models) is not list:
            self.models = [self.models]
        self.all_evaluations = []
        self.evaluation = dict.fromkeys(self.data_list)
        self.loaded_model = 0
        for model, pred_str in zip(self.models, self.pred_str_list):
            self.selected_sensors = model.selected_sensors
            self.pred_str = pred_str
            for dataset in self.data_list:
                benchmark = self.load_benchmark_data(dataset)
                result = self.evaluator(model, benchmark[0], benchmark[1])
                self.evaluation[dataset] = copy.deepcopy(result)
            self.all_evaluations.append(copy.deepcopy(self.evaluation))
            self.loaded_model += 1
        pickle.dump(self.all_evaluations, open(self.save_prefix + ".all_evaluations",'wb'))
        #self.all_evaluations = pickle.load(open(self.save_prefix + ".all_evaluations",'rb'))
        print("printing acc tbl")
        self.generate_acc_tbl()
        print("printing acc acc_plt")
        self.generate_avg_acc_plt()
        print("printing acc cfm")
        self.generate_confusion_matrix()
        print("printing acc ss")
        self.generate_sensor_selection()
        print("printing stat tbl")
        self.generate_stat_tbl()
        print("Complete")
    def generate_stat_tbl(self):
        if self.mode == "regression":
            fname_list = [basename(dataset)[:-8] + ".stat_tbl" + ".csv" for dataset in self.data_list]
            columns = ["Mean_Squared_Error", "Standard_Deviation_Error", "R^2"]
            for model_eval, pred_str in zip(self.all_evaluations, self.pred_str_list):
                row_label = self.exp_name + ".pred_str." + str(pred_str)
                for data_eval, fname in zip(model_eval.values(), fname_list):
                    data = [data_eval[key] for key in columns]
                    df = pandas.DataFrame(np.array(data).reshape(1,-1), index=[row_label], columns=columns)
                    with open(fname, 'a') as f:
                        df.to_csv(f, header=f.tell()==0)
    def generate_avg_acc_plt(self):
        """Generate average accurary plot to compare the performance of multiple models.
        """
        if self.mode == "regression":
            model_count = len(self.all_evaluations)
            avg_acc = np.array([result["acc"] for evaluation in self.all_evaluations for result in evaluation.values()]).reshape(len(self.all_evaluations[0]),-1)
            if avg_acc.ndim > 1:
                avg_acc = np.mean(avg_acc, axis=0)
            pred_str = np.array(self.pred_str_list)

            # calculating the trend line
            from sklearn.linear_model import Ridge
            lr = Ridge()
            lr.fit(pred_str.reshape(-1,1), avg_acc.reshape(-1,1))
            common_x = pred_str.flatten()
            bar_y = avg_acc.flatten()
            curve_y = (lr.coef_*pred_str+lr.intercept_).flatten()
            plt.bar(common_x, bar_y)
            plt.plot(common_x, curve_y, color='orange')
            for x, y in zip(common_x, curve_y):\
                plt.annotate("{:5.2f}".format(y), (x,y))
            plt.xlabel("Prediction Capability")
            plt.ylabel("Regression Accurary")
            #tikzplotlib.save(self.latex_fig.joinpath(self.save_prefix+".avg_acc.tex"))
            plt.savefig(self.latex_fig.joinpath(self.save_prefix+".avg_acc.pdf"))
            plt.close()
    def generate_acc_tbl(self):
        fname = "overall_acc_tbl" + ".csv"
        pred_str_5_evaluation = self.all_evaluations[self.index_5]
        all_y = [result["acc"] for result in pred_str_5_evaluation.values()]
        df = pandas.DataFrame(np.array(all_y).reshape(1,-1), index=[self.model_fname], columns=[self.data_list])
        with open(fname, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
    def generate_sensor_selection(self):
        num = 1
        for model in self.models:
            fname = self.latex_fig.joinpath(self.save_prefix+".selected_sensor_grid" + str(num)+".pdf")
            num += 1
            sensors = model.selected_sensors
            dim = len(sensors)
            row = int(np.sqrt(dim))
            sensor_map = sensors.reshape((row, row))
            # rotating the map for view
            sensor_map = np.flip(sensor_map, axis=0)

        # prepare for plotting flp
            # read flp img
            flp = plt.imread(str(self.flp))
            # set up alpha channel for tranparency
            alphas = Normalize(0, .3, clip=True)(np.abs(np.sum(flp, axis=-1)))
            alphas = np.ones_like(alphas) - np.clip(alphas, .6, 1)  # alpha value clipped at the bottom at .4
            flp = np.dstack([flp, alphas])
        # prepare for plotting selected sensor
            # create rgb space for hosting the sensor map
            drawing = np.ones(shape=(sensor_map.shape)+(3,))
            # set selected sensor to be red
            drawing[:,:,0] = sensor_map
            drawing[:,:,1] = np.bitwise_not(sensor_map)
            drawing[:,:,2] = np.bitwise_not(sensor_map)
            # resize sensor map
            drawing = cv2.resize(drawing, dsize=flp.shape[:2], interpolation=cv2.INTER_NEAREST)
        # ploting
            plt.imshow(drawing)
            plt.imshow(flp)
            #tikzplotlib.save(fname)
            # plt.tick_params(
            #     axis='both',          
            #     which='both',      # both major and minor ticks are affected
            #     bottom=False,      # ticks along the bottom edge are off
            #     top=False,         # ticks along the top edge are off
            #     left=False,
            #     right=False,
            #     labelbottom=False) # labels along the bottom edge are off
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False) 
            plt.savefig(fname)
            plt.close()
    def generate_confusion_matrix(self):
        for model_eval, pred_str in zip(self.all_evaluations, self.pred_str_list):
            cfm = np.zeros(shape=(2,2))
            for benchmark_result in model_eval.values():
                cfm[0,0] += benchmark_result["tp"]
                cfm[0,1] += benchmark_result["fp"]
                cfm[1,0] += benchmark_result["fn"]
                cfm[1,1] += benchmark_result["tn"]
            cfm_df = pd.DataFrame(cfm,index=["Positive", "Negative"],columns=["Positive", "Negative"])      
            # cfm_df.index.name = 'Actual'
            # cfm_df.columns.name = 'Predicted'
            p=pretty_plot_confusion_matrix(cfm_df)      
            plt.savefig(self.latex_fig.joinpath(self.exp_name + ".pred_str." + str(pred_str) + ".confusion_matrix.pdf"))
            p.close()
    def load_benchmark_models(self, model_fname):
        #self.models = []
        #for fname in model_list:
        if isinstance(model_fname, list): 
            saved_model = voltnet_model()
            saved_model.load()
        else:
            saved_model = pickle.load(open(model_fname, 'rb'))
        self.models= saved_model
        print(self.models)
        return self.models
    def load_benchmark_data(self, fname):
        """ loaded data should have shapes (samples, nodes)"""
        if self.mode == "regression":
            if fname.endswith(".gridIR"):
                loader = regression_training_data_factory([fname], lines_to_read=self.lines_to_read)
                [x, tag] = loader.generate()
                tag = np.array(tag)
                x = x.T
            if fname.endswith(".h5"):
                with h5py.File(fname, 'r') as f: # loading volnet files
                    x = f["x"][self.lines_to_jump:self.lines_to_read+self.lines_to_jump,:,-self.pred_str,-1]
                    tag = f["y"][self.lines_to_jump:self.lines_to_read+self.lines_to_jump]    
        if self.mode == "classification":
            if fname.endswith(".h5"): # loading distribution files
                with h5py.File(fname, 'r') as f:
                    x = f["x"][self.lines_to_jump:self.lines_to_read+self.lines_to_jump,:]
                    tag = f["y"][self.lines_to_jump:self.lines_to_read+self.lines_to_jump]      
        return [x, tag.astype('int')]
    def benchmark_from_ckp(self, ckp_list):
        for ckp in ckp_list:
            self.all_evaluations = pickle.load(open(ckp,'rb'))
            self.generate_acc_tbl()
            self.generate_avg_acc_plt()
            self.generate_confusion_matrix()
            self.generate_sensor_selection()
if __name__ == "__main__":
    if os.name == "nt":
        core = 2
        if core == 2:
            flp = Path(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("2c.png")
        elif core == 4:
            flp = Path(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("4c.png")
        elif core == 16:
            flp = Path(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("16c.png")
        f_list = [
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "blackscholes2c" + ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "bodytrack2c" + ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "freqmine2c"+ ".gridIR",
        "C:\\Users\\Yi\Desktop\\analysis_pred\\pyscripts\\" + "facesim2c"+ ".gridIR",
        ]
    else:
        core = 2
        if core == 2:
            flp = Path(r".").joinpath("2c.png")
        elif core == 4:
            flp = Path(r".").joinpath("4c.png")
        elif core == 16:
            flp = Path(r".").joinpath("16c.png")
        f_list = [
        "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
        "/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR",
        "/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR",
        "/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR",
        ]
    lines_to_read = 1000
    lines_to_jump = 10000
    f_list = ["F:\\lstm_data\\VoltNet_2c.str0.h5"]
    #gp_models = "gl.model" 
    pred_str_list = [0,5,10,20,30]
    #gp_models = "gl.pred_str.models" 
    gp_models = "gl.pred_str.models.test.mean.cluster" 
    gp_benchmark = benchmark_factory(gp_models, f_list,flp=flp, exp_name="gp",mode="regression",
                                     pred_str_list=pred_str_list, lines_to_read=lines_to_read, lines_to_jump=lines_to_jump)
    gp_benchmark.benchmarking()
    #gp_benchmark.benchmark_from_ckp(ckp_list=["gp.regression.all_evaluations"])

    ee_models = "ee.original.pred_str.model"
    ee_benchmark = benchmark_factory(ee_models, f_list,flp=flp, exp_name="ee.original",mode="regression",
                                     pred_str_list=pred_str_list, lines_to_read=lines_to_read, lines_to_jump=lines_to_jump)
    ee_benchmark.benchmarking()    
    #ee_benchmark.benchmark_from_ckp(ckp_list=["ee.regression.all_evaluations"])
    ee_models = "ee.segmented.pred_str.model"
    ee_benchmark = benchmark_factory(ee_models, f_list,flp=flp, exp_name="ee.segmented",mode="regression",
                                     pred_str_list=pred_str_list, lines_to_read=lines_to_read, lines_to_jump=lines_to_jump)
    #ee_benchmark.benchmark_from_ckp(ckp_list=["ee.regression.all_evaluations"])
    ee_benchmark.benchmarking()

    # vn_models = ["voltnet"]
    # f_list = ['F:\\lstm_data\\prob_distribution.h5']
    # vn_benchmark = benchmark_factory(vn_models, f_list, flp=flp, exp_name="vn.test", mode="classification", pred_str_list=[0], lines_to_read=10000, lines_to_jump=10000)
    # vn_benchmark.benchmarking()
    print("complete")
    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)