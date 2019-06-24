import numpy as np
import operator
from loading import read_volt_grid
def get_violation(data, occurrence=np.array([],dtype=np.double), ref=1, thres=4)->np.ndarray:
    """Get the coordination of voltage violation node 
    
    Arguments:
        data {1D/2D np array} -- array from loading.py. If 2D, square grids are stored in columns witch are ordered in time.
        thres {int} -- a percentage defines the margin for threshold
        ref {int} -- the reference (expected) voltage 
    Returns:
        np.ndarray -- The info of each violation is stored in a column.
                      Within the column, the info is ordered as 
                      [x coordinate, y coordinate, column number in "data", value of the node]
                      note, x coor is the column number, y coor is the row number
    """
    gird_size = data.shape[0]
    dim = int(np.sqrt(gird_size)) # This has to be a square gird
    margins = np.array([1 + thres/100, 1 - thres/100]) * ref
    time_stamp = 0
    new_occurrence = []
    for gird in data.T:
        higher = gird > margins[0]
        lower = gird < margins[1]
        vios = np.bitwise_or(higher, lower)
        serialized_coor = vios.nonzero()[0] #vios is 1 d, but nonzero still return a 2d array
        
        x_coor = serialized_coor % dim
        y_coor = serialized_coor // dim 
        volt = gird[serialized_coor]
        stamps = [time_stamp] * len(serialized_coor)

        current_report = []
        current_report.append(x_coor)
        current_report.append(y_coor)
        current_report.append(volt)
        current_report.append(stamps)

        new_occurrence.append(current_report)
        time_stamp += 1
    if current_report[0].size:
        new_occurrence = np.hstack(new_occurrence)
    if occurrence.size:
        if current_report[0].size:
            occurrence = np.hstack((occurrence, new_occurrence))
    else:
        if not current_report[0].size:
            new_occurrence =  np.array([],dtype=np.double)
        occurrence = new_occurrence
    return occurrence

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


if __name__ == "__main__":
    #a = np.array([[1,0.99,2,0],[1,1,1,1],[1,0.99,2,0]])


    gridIR = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    maxlen = 5000
    start_line = 10000
    occurrence = np.array([],dtype=np.double)
    for index in range(1,34):
        x_train = read_volt_grid(gridIR, maxlen, maxlen * index)
        occurrence = get_violation(x_train, occurrence,thres=1)
        print(index)

    placement = eagle_eye(occurrence, 10)
    print(occurrence)
    print(placement)
