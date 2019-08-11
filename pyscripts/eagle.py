import numpy as np
import operator
from loading import *
from flp import *



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
    def __init__(self, flp_fname, placement_plan, segment_trigger=True):
         self.flp_fname = flp_fname
         self.placement_plan = placement_plan
         # grid_size
         #self.flp = self.load_flp(self.flp_fname)
         self.flp = flp
         self.segment_trigger = segment_trigger
         self.placement = []
    def train(self, training_data):
        self.occurrence = training_data
    def eagle_eye(self, budget)->list:
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
        if not self.occurrence.size:
            return []
        hashT = dict()
        # This is NOT a proper implementation. But close enough for our case
        for vio in self.occurrence.T:
            key = "{:01g},{:01g}".format(vio[0],vio[1])
            if key in hashT:
                hashT[key] += 1
            else:
                hashT[key] = 1
        for index in range(budget):
            candidate = max(hashT.items(),key=operator.itemgetter(1))[0]
            self.placement.append(candidate)
            del hashT[candidate]
        return self.placement
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
        pitch = width / rows
        # length = rows * columns
        unscaled_mask = np.zeros((rows, columns))
        if self.segment_trigger:
            for unit in flp:
                umap['placement_plan'].append((unit, index, 1))
                go_right = int(flp[unit][0] // pitch)
                go_up = int(flp[unit][1] // pitch)
                #upper left corner
                x = int(flp[unit][2] // pitch)
                y = int(rows - flp[unit][3] // pitch - go_up)
                unscaled_mask[y:y+go_up, x:x+go_right] = index
                index += 1    
        else:
            unscaled_mask[:,:] = 1
        #umap['mask'] = unscaled_mask.flatten()
        umap['mask'] = unscaled_mask
        return umap
    def flp_filter(self, occurrence, umap, unit_digit):
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
            row = int(umap['meta'][0] - vio[1] - 1)
            col = int(vio[0] - 1)
            if int(umap['mask'][row, col]) == unit_digit:
                segmented.append(vio.reshape(-1, 1))
        if segmented:
            segmented = np.hstack(segmented)
            return segmented
        return np.array([],dtype=np.double)
class ee_model():
    def __init__(self, flp_fname, gridIR, segment_trigger=True, placement_mode="uniform", placement_para=46):
        self.flp_fname = flp_fname
        self.girdIR = gridIR
        self.segment_trigger = segment_trigger
        self.sensor_distribution = distribution
        self.total_budget = total_budget
        if placement_mode == "uniform":
            self.total_budget = placement_para
            self.placer = self.uniform_placement
        # loading
        self.flp = self.load_flp()
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
    def init_selector(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.selector = ee_sensor_selector(alpha=alpha,fit_intercept=fit_intercept,
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
    def generate_placement_plan(self):
        placement_plan = []
        index = 1
        if self.segment_trigger:
            for unit in self.flp:
                budget = self.placer(unit)
                placement_plan((unit, index, budget))
                if distribution == "uniform":
                    placement_plan.append((unit, index, 1))
                index += 1   
        else:
            placement_plan = [("all", index, total_budget)]
    def uniform_placement(self, unit, index):
        
        return budget

if __name__ == "__main__":

    gridIR = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    maxlen = 5000
    start_line = 10000
    (occurrence, dim) = read_violation(gridIR, 100000, trace=10)

    fname = r"C:\Users\Yi\Desktop\Software reference\VoltSpot-2.0\example.flp"
    flp = load_flp(fname)
    umap = get_mask(flp, dim)
    
    result = []
    for unit in umap['placement_plan']:
        if unit[2]:
            segmented = flp_filter(occurrence, umap, unit[1])
            result = eagle.eagle_eye(segmented, unit[2], result)

    print(result)
