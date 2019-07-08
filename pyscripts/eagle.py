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


if __name__ == "__main__":

    gridIR = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    maxlen = 5000
    start_line = 10000
    (occurrence, dim) = read_violation(gridIR, 100000, trace=10)

    fname = r"C:\Users\Yi\Desktop\Software reference\VoltSpot-2.0\example.flp"
    flp = load_flp(fname)
    umap = get_mask(flp, dim)
    
    result = []
    for unit in umap['decoder']:
        if unit[2]:
            segmented = flp_filter(occurrence, umap, unit[1])
            result = eagle.eagle_eye(segmented, unit[2], result)

    print(result)
