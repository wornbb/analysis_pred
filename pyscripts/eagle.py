import numpy as np
import operator
from loading import *
from flp import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import PureWindowsPath
def eagle_eye(occurrence, budget ,placement=[])->list:
    """vanilla eagle eye algorithm. 
    
    Arguments:
        occurrence {2d array} -- The info of each violation is stored in a column.
                      Within the column, the info is ordered as 
                      [x coordinate, y coordinate, column number in "data", value of the node]
                      note, x coor is the column number, y coor is the row number
        budget {int} -- number of sensors to be placed
    
    Returns:
        list -- placement of sensors. Coordiantes are stored in pairs. Pairs are stacked vertically. 
                      For each pair, first column is the x coor, and second is y coor
    """
    if not occurrence.size:
        return []
    hashT = dict()
    # This is NOT a proper implementation. But close enough for our case
    for vio in occurrence.T:
        key = "{:01g},{:01g}".format(vio[0],vio[1])
        if key in hashT:
            hashT[key] += 1
        else:
            hashT[key] = 1
    
    for index in range(budget):
        candidate = max(hashT.items(),key=operator.itemgetter(1))[0]
        placement.append(candidate)
        del hashT[candidate]
    return placement

class ee_sensor_selector():
    def __init__(self, flp, placement_plan, grid_size, segment_trigger=True):
         self.placement_plan = placement_plan
         self.grid_size = grid_size
         self.flp = flp
         self.segment_trigger = segment_trigger
         self.raw_placement = []
         self.placement = np.zeros(grid_size, dtype=bool)
    def train(self, training_data):
        self.umap = self.get_mask(self.flp, self.grid_size)
        for unit_plan in self.placement_plan:
            if unit_plan[2]:
                segmented = self.flp_filter(training_data, unit_plan[1])
                self.raw_placement = self.eagle_eye(segmented, unit_plan)
    def eagle_eye(self, occurrence, unit_plan)->list:
        """vanilla eagle eye algorithm. 
        Arguments:
            occurrence {2d array} -- The info of each violation is stored in a column.
                            Within the column, the info is ordered as 
                            [x coordinate, y coordinate, column number in "data", value of the node]
                            note, x coor is the column number, y coor is the row number
            budget {int} -- number of sensors to be placed
        
        Returns:
            list -- placement of sensors. Coordiantes are stored in pairs. Pairs are stacked vertically. 
                            For each pair, first column is the x coor, and second is y coor
        """
        unit = unit_plan[0]
        budget = unit_plan[2]
        if not occurrence.size:
            for index in range(budget):
                unit_flp = self.flp[unit]
                minx = int(unit_flp[2] // self.pitch)
                x_range = int(unit_flp[0] // self.pitch)
                miny = int(unit_flp[3]// self.pitch)
                y_range = int(unit_flp[1]// self.pitch)
                if x_range == 0:
                    x_range = 1
                if y_range == 0:
                    y_range =1
                x = np.random.randint(minx, minx + x_range)
                y = np.random.randint(miny, miny + y_range)
                key = "{:01g},{:01g}".format(y,x)
                self.raw_placement.append(key)
        else:
            hashT = dict()
            # This is NOT a proper implementation. But close enough for our case
            for vio in occurrence.T:
                key = "{:01g},{:01g}".format(vio[0],vio[1])
                if key in hashT:
                    hashT[key] += 1
                else:
                    hashT[key] = 1
            for index in range(budget):
                candidate = max(hashT.items(),key=operator.itemgetter(1))[0]
                self.raw_placement.append(candidate)
                del hashT[candidate]
        return self.raw_placement
    def get_mask(self, flp, dim)->dict:
        """Transform a flp in dictionary to a bit map in a vector. 
        The vector is has the same dimension as the "grid vector" read from gridIR

        flp: {name: [<width>,<height>,<left-x>,<bottom-y>]}
        Arguments:
            flp {dict} -- Floorplan generated from get_flp
            dim {int} -- dimension of the "grid vector"
        
        Returns:
            dict -- mask: a vector indicating which element belongs to which unit
                    placement_plan: a list of tuples (unit name, digit for unit, number of sensor for this unit)
                    meta: a list [total rows, total columns]
        """
        index = 1
        umap = dict(mask=[], meta=[])
        # get the total width 
        x_min = min(flp.values(), key=operator.itemgetter(2))
        x_max = max(flp.values(), key=operator.itemgetter(2))
        width = x_max[2] - x_min[2] + x_max[0]
        # # get the total height
        # y_min = min(flp.values(), key=operator.itemgetter(3))
        # y_max = max(flp.values(), key=operator.itemgetter(3))
        # height = y_max[3] - y_min[3] + y_max[1]

        # assume square flp
        rows = int(np.sqrt(dim) )
        columns = rows
        umap['meta'] = [rows, columns]
        self.pitch = width / rows
        # length = rows * columns
        unscaled_mask = np.zeros((rows, columns))
        if self.segment_trigger:
            for unit in flp:
                go_right = int(flp[unit][0] // self.pitch)
                go_up = int(flp[unit][1] // self.pitch)
                #upper left corner
                x = int(flp[unit][2] // self.pitch)
                y = int(rows - flp[unit][3] // self.pitch - go_up)
                unscaled_mask[y:y+go_up, x:x+go_right] = index
                index += 1    
        else:
            unscaled_mask[:,:] = 1
        #umap['mask'] = unscaled_mask.flatten()
        umap['mask'] = unscaled_mask
        return umap
    def flp_filter(self, occurrence, unit_digit):
        """segment the occurrence based on flp
        
        Arguments:
            occurrence {np.ndarray} -- The info of each violation is stored in a column.
                        Within the column, the info is ordered as 
                        [x coordinate, y coordinate, column number in "data", value of the node]
                        note, x coor is the column number, y coor is the row number
            umap {dict} -- mask: a vector indicating which element belongs to which unit
                            decoder: a list of tuples (unit name, digit for unit, number of sensor for this unit)
                    meta: a list [total rows, total columns]

            unit_digit {int} -- the number assigned to present the unit
        """
        segmented = []
        for vio in occurrence.T:
            row = int(self.umap['meta'][0] - vio[1] - 1)
            col = int(vio[0] - 1)
            if int(self.umap['mask'][row, col]) == unit_digit:
                segmented.append(vio.reshape(-1, 1))
        if segmented:
            segmented = np.hstack(segmented)
            return segmented
        return np.array([],dtype=np.double)
    def predict(self):
        for node in self.raw_placement:
            [row, col] = np.fromstring(node, dtype=int, sep=',')
            self.placement[row *int(np.sqrt(self.grid_size)) + col] = True
        return self.placement
class ee_model():
    def __init__(self, flp_fname, gridIR, pred_str=1, segment_trigger=True, placement_mode="uniform", placement_para=1):
        self.flp_fname = flp_fname
        self.girdIR = gridIR
        self.pred_str = pred_str
        self.segment_trigger = segment_trigger
        self.placement_mode = placement_mode
        self.placement_para = placement_para
        if placement_mode == "uniform":
            self.single_budget = placement_para
            self.placer = self.uniform_placement
        # loading
        self.flp = self.load_flp()
        [self.training_data, self.grid_size] = read_violation(gridIR, lines_to_read=6000, trace=10)
        # init key components
        self.placement_plan = self.generate_placement_plan()
        self.selector = ee_sensor_selector(self.flp, self.placement_plan, self.grid_size, self.segment_trigger)
        self.init_predicor()
    def load_flp(self)->dict:
        """load the floorplan into a dictionary
        
        Arguments:
            fname {string} -- file path
        
        Returns:
            np.ndarray -- A dictionary with keys as unit name. The value is:
                            [<width>,<height>,<left-x>,<bottom-y>]
        """
        flp = dict()
        with open(self.flp_fname, 'r') as f:
            for line in f:
                if '#' in line:
                    pass
                elif line.rstrip('\n'):
                    unit = line.split()
                    flp[unit[0]] = np.array(unit[1:], dtype=np.float64)
        return flp
    def init_predicor(self):
        self.predictor = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=5, n_jobs=6)
    def fit(self, data):
        # sensor selection
        self.selector.train(self.training_data)
        self.selected_sensors = self.selector.predict()
        # data filtering
        self.selected_x = data[self.selected_sensors,:-self.pred_str].T
        y = data[np.bitwise_not(self.selected_sensors),self.pred_str:].T 
        self.predictor.fit(X=self.selected_x, y=y)
    def predict(self, x):
        x = x[:,self.selected_sensors]
        return self.predictor.predict(x)
    def evaluate(self, x, y):
        x = x[:,self.selected_sensors]
        y_pred = self.predictor.predict(x)
        return [0, mean_squared_error(y, y_pred)]
    def generate_placement_plan(self):
        placement_plan = []
        index = 1
        if self.segment_trigger:
            for unit in self.flp:
                budget = self.placer(unit, index)
                placement_plan.append((unit, index, budget))
                index += 1   
        else:
            total_budget = self.single_budget * len(self.flp)
            placement_plan = [("all", index, total_budget)]
        return placement_plan
    def uniform_placement(self, unit, index):
        return self.single_budget
    def show_flp(self):
        row = int(np.sqrt(self.grid_size))
        flp_img = np.zeros((row,row))
        color = 1
        for unit_flp in self.flp.values():
            minx = int(unit_flp[2] // self.selector.pitch)
            x_range = int(unit_flp[0] // self.selector.pitch)
            miny = int(unit_flp[3]// self.selector.pitch)
            y_range = int(unit_flp[1]// self.selector.pitch)
            if x_range == 0:
                x_range = 1
            if y_range == 0:
                y_range =1
            row = np.random.randint(minx, minx + x_range)
            col = np.random.randint(miny, miny + y_range)
            flp_img[miny:miny+y_range,minx:minx+x_range] = color
            color += 1
        plt.imshow(np.flip(flp_img, axis=0))
        plt.colorbar()
        plt.show() 
    def show_sensors(self):
        row = int(np.sqrt(self.grid_size))
        grid = self.selected_sensors.reshape((row,row))
        plt.imshow(np.flip(grid, axis=0).astype(int))
        plt.colorbar()
        plt.show()
if __name__ == "__main__":
    core = 2
    if core == 2:
        fname = PureWindowsPath(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("Penryn22_ruby_ya_2c_v13.flp")
    elif core == 4:
        fname = PureWindowsPath(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("Penryn22_ruby_ya_4c_v13.flp")
    elif core == 16:
        fname = PureWindowsPath(r"C:\Users\Yi\Desktop\analysis_pred\pyscripts").joinpath("Penryn22_ruby_ya_16c_v13.flp")
    gridIR = "F:\\Yaswan2c\\Yaswan2c.gridIR"
    data = read_volt_grid(gridIR, 40)
    models = []
    for pred_str in range(5):
        pred_str += 1
        ee_test = ee_model(flp_fname=fname, gridIR=gridIR, pred_str=pred_str)
        ee_test.fit(data)
        models.append(ee_test)
    import pickle
    pickle.dump(models, open("ee.model", "wb"))
    # maxlen = 5000
    # start_line = 10000
    # (occurrence, dim) = read_violation(gridIR, 100000, trace=10)

    # fname = r"C:\Users\Yi\Desktop\Software reference\VoltSpot-2.0\example.flp"
    # flp = load_flp(fname)
    # umap = get_mask(flp, dim)
    
    # result = []
    # for unit in umap['placement_plan']:
    #     if unit[2]:
    #         segmented = flp_filter(occurrence, umap, unit[1])
    #         result = eagle.eagle_eye(segmented, unit[2], result)

    # print(result)
