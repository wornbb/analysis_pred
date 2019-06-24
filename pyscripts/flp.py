import numpy as np
import operator
import eagle
import loading
def load_flp(fname)->dict:
    """load the floorplan into a dictionary
    
    Arguments:
        fname {string} -- file path
    
    Returns:
        np.ndarray -- A dictionary with keys as unit name. The value is:
                        [<width>,<height>,<left-x>,<bottom-y>]
    """
    flp = dict()
    with open(fname, 'r') as f:
        for line in f:
            if '#' in line:
                pass
            elif line.rstrip('\n'):
                unit = line.split()
                flp[unit[0]] = np.array(unit[1:], dtype=np.float64)
    return flp
def get_mask(flp, dim)->dict:
    """Transform a flp in dictionary to a bit map in a vector. 
    The vector is has the same dimension as the "grid vector" read from gridIR

    flp: {name: [<width>,<height>,<left-x>,<bottom-y>]}
    Arguments:
        flp {dict} -- Floorplan generated from get_flp
        dim {int} -- dimension of the "grid vector"
    
    Returns:
        dict -- mask: a vector indicating which element belongs to which unit
                decoder: a list of tuples (unit name, digit for unit, number of sensor for this unit)
                meta: a list [total rows, total columns]
    """
    index = 1
    mask = np.zeros(dim)
    umap = dict(mask=[], decoder=[], meta=[])
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
    length = rows * columns
    unscaled_mask = np.zeros((rows, columns))
    for unit in flp:
        umap['decoder'].append((unit, index, 1))
        go_right = int(flp[unit][0] // pitch)
        go_up = int(flp[unit][1] // pitch)
        #upper left corner
        x = int(flp[unit][2] // pitch)
        y = int(rows - flp[unit][3] // pitch - go_up)
        unscaled_mask[y:y+go_up, x:x+go_right] = index
        index += 1    
    #umap['mask'] = unscaled_mask.flatten()
    umap['mask'] = unscaled_mask
    return umap
def flp_filter(occurrence, umap, unit_digit):
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


if __name__ == "__main__":

    gridIR = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    maxlen = 5000
    start_line = 10000
    occurrence = np.array([],dtype=np.double)
    for index in range(100000):
        x_train = loading.read_volt_grid(gridIR, maxlen, maxlen * index)
        occurrence = eagle.get_violation(x_train, occurrence, thres=5)    

    fname = r"C:\Users\Yi\Desktop\Software reference\VoltSpot-2.0\example.flp"
    flp = load_flp(fname)
    umap = get_mask(flp, x_train.shape[0])
    
    result = []
    for unit in umap['decoder']:
        if unit[2]:
            segmented = flp_filter(occurrence, umap, unit[1])
            result = eagle.eagle_eye(segmented, unit[2], result)

    print(result)

