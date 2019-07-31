import h5py
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from collections import deque
def read_volt_grid(file, lines_to_read, start_line = 0):
    """Read .gridIR files and store them in a 2D array
    
    Arguments:
        file {string} -- complete file name with directory unless in the same root folder
        lines_to_read {int} -- How many lines do we want to read?
    
    Keyword Arguments:
        start_line {int} -- jump a few lines ahead (default: {0})
    
    Returns:
        2D np array -- All data read. The grid is stored in a column.
    """
    dim_col = 1
    batch = []
    with open(file, 'r') as v:
        for i in range(start_line):
            v.readline()
        for i in range(start_line, start_line + lines_to_read):
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                batch.append(v_formated)
            
            # if i == start_line:
            #     batch = v_formated
            # else:
            #     if v_formated.shape[0] == 0:
            #         print(i)
            #         print(vline)
    if batch:
        batch = np.column_stack(batch)
    return batch
def get_violation(data, occurrence=np.array([],dtype=np.double), ref=1, thres=4, prev=[], mode=0, reverse=False)->np.ndarray:
    """Get the coordination of voltage violation node from loaded grid array.
    
    Arguments:
        data {1D/2D np array} -- array from loading.py. If 2D, square grids are stored in columns witch are ordered in time.
        thres {int} -- a percentage defines the margin for threshold
        ref {int} -- the reference (expected) voltage 
        prev {np 2darray} -- a collection of previous CPU cycles. Only used when a trace is needed
        mode {int} -- When mode = 0, it only returns the violation point at a certain time
                      When mode = n where n is not 0, it returns a voltage trace with length of n time points.
        reverse {bool} -- Default: False. When set to true, it will return the coordinate of non-voltage violation node instead
    Returns:
        np.ndarray -- The info of each violation is stored in a column.
                      Within the column, the info is ordered as 
                      [x coordinate, y coordinate, column number in "data", value of the node]
                      note, x coor is the column number, y coor is the row number
                   -- In the case of Trace mode, the column of output will be:
                      [x coor, y coor, column number in "data", values of the trace with the most recent value on top]
    """
    gird_size = data.shape[0]
    dim = int(np.sqrt(gird_size)) # This has to be a square gird
    margins = np.array([1 + thres/100, 1 - thres/100]) * ref
    time_stamp = 0
    new_occurrence = []
    # following trick deals with the case when data is just a vector
    ndim = data.ndim
    if ndim == 1:
        data = [data]
    else:
        data = data.T
    for gird in data:
        higher = gird > margins[0]
        lower = gird < margins[1]
        vios = np.bitwise_or(higher, lower)
        if reverse:
            vios = np.bitwise_not(vios)
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

        #new_occurrence.append(current_report)
        time_stamp += 1
    if current_report[0].size:
        #new_occurrence = np.hstack(new_occurrence)
        new_occurrence = np.row_stack(current_report)
        if mode:
            trace = prev[:,vios]
            new_occurrence = np.vstack((new_occurrence, trace))

    if occurrence.size:
        if current_report[0].size:
            occurrence = np.hstack((occurrence, new_occurrence))
    else:
        if not current_report[0].size:
            new_occurrence =  np.array([],dtype=np.double)
        occurrence = new_occurrence
    return occurrence

def read_violation(file, lines_to_read=0, start_line=0, trace=0, thres=4, ref=1, count=0, reverse=False):
    """Read the gridIR file but only return the occurrence when there is a violation.
    
    Arguments:
        file {str} -- file path
    
    Keyword Arguments:
        lines_to_read {int} -- How many lines (CPU cycles) to read
        start_line {int} -- Jump start. How many lines to skip before read (default: {0})
        trace {int} -- Instead of a violation point, we get a violation trace in hope to find a pattern. 
                        When a violation occurs, how many previous cycles we want to store as well. (default: {0})
        thres {int} -- threshold in percentage (default: {4})
        ref {int} -- reference voltage level (normal voltage level)(default: {1})
        reverse {bool} -- Default: False. When set to true, it will return the coordinate of non-voltage violation node instead
        count {int} -- Default: np.inf. Program stops once the loaded occurrence reaches the count.
    Returns:
        (np.ndarray, int) -- A tuple where:
                            - The first: actual violation occurrence with format:
                                * The info of each violation is stored in a column.
                                Within the column, the info is ordered as 
                                [x coordinate, y coordinate, column number in "data", value of the node]
                                note, x coor is the column number, y coor is the row number
                                * In the case of Trace mode, the column of output will be:
                                [x coor, y coor, column number in "data", values of the trace with the most recent value on top]
                            - The Second: The length of grid vector

    """
    # preprocessing key arguments. So that the input type remains as int.
    if not lines_to_read:
        lines_to_read = np.inf
    if not count:
        count = np.inf
    batch = []
    total = count
    with open(file, 'r') as v:
        for i in range(start_line):
            v.readline()
        # buffer = deque()
        buffer = [] # buffer is a queue with length trace
        #fill que
        for i in range(trace-1):
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                buffer.append(v_formated)
            else:
                print("this is a very unlikely situation. Why would there be a blank line in the file?")
        buffer = np.array(buffer)
        while lines_to_read > 0:
            if vline == '':
                break
            lines_to_read -= 1
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                vios = get_violation(v_formated, prev=buffer, mode=trace, reverse=reverse)
                buffer = np.roll(buffer, -1, axis=0)
                buffer[-1,:] = v_formated
                if vios.size:
                    batch.append(vios)
                    count -= 1
                    if count == 0:
                        break
    if batch:
        batch = np.column_stack(batch)
    dim = v_formated.shape[0]
    if count > 0:
        print("Warning: File ends before enough instances collected. Total counts:", total - count)
    return (batch, dim)
def generate_prediction_data(file, lines_to_read=0, selected_sensor=[], trace=39, pred_str=5, thres=4, ref=1, global_vio=True):
    """Generate the voltage trace only at selected sensors. Generated trace batch will be balanced with violaions and normal.

    Arguments:
        file {str} -- gridIR name

    Keyword Arguments:
        lines_to_read {int} -- [description] (default: {0})
        selected_sensor {list} -- mask for selecting sensors (default: {[]})
        trace {int} -- length of the trace (default: {20}) The exact lenght = trace - pred_str
        pred_str {int} -- how many cpu cycles ahead to predict (default: {5})
        thres {int} -- [description] (default: {4})
        ref {int} -- [description] (default: {1})
        global_vio {bool} -- determine whether the violation check is global or not.
                            if true: traces are recorded as long as there is a violation on the grid.
                            if false: traces are only recorded if violation happens at the selected nodes(default: {True})

    Returns:
        (batch, tag) -- batch: traces
                        tag:   violation types:
                                    0: no violation
                                    1: local violation
                                    2: global violation
    """
    if not lines_to_read:
        lines_to_read = np.inf
    count = 0
    batch = []
    tag = []
    with open(file, 'r') as v:

        buffer = [] # buffer is a queue with length trace
        #fill que
        for i in range(trace):
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                buffer.append(v_formated)
            else:
                print("this is a very unlikely situation. Why would there be a blank line in the middle of the file?")
        buffer = np.array(buffer) # make buffer a np array to fasten the operation
        if selected_sensor == "all":
            global_vio = False # disable tag=2
            selected_sensor = np.ones_like(v_formated, dtype=bool)
        counter_index = 0
        timer_index = 1
        norm_counter = 0
        norm_timer = 0
        norm = [norm_counter, norm_timer]
        local_vios = np.array([])
        vios = np.array([])
        while lines_to_read > 0:
            lines_to_read -= 1
            vline = v.readline()
            if vline == '':
                break
            v_formated = np.fromstring(vline, sep='    ', dtype='float')

            if v_formated.size:
                if not global_vio:
                    local_vios = get_violation(v_formated[selected_sensor], prev=buffer, mode=trace, ref=ref, thres=thres)
                else:
                    vios = get_violation(v_formated, prev=buffer, mode=trace, ref=ref, thres=thres)
                    
                # update buffer like a queue. queue is too slow for other operation
                buffer = np.roll(buffer, -1, axis=0)
                buffer[-1,:] = v_formated
                # logic to process the grid
                if vios.size and not local_vios.size: # violation happens globally but not locally
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str - 1
                    batch.append(buffer[:trace-pred_str, selected_sensor].T)
                    tag.append(1)
                elif local_vios.size: # local violation
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str - 1
                    batch.append(buffer[:trace-pred_str, selected_sensor].T)
                    tag.append(2)
                else: # normal
                    if norm[counter_index] != 0: # only update timer if there is a counter.
                        norm[timer_index] -= 1
                if norm[timer_index] % (trace + pred_str)==0 and norm[counter_index] != 0:
                    norm[counter_index] -= 1
                    batch.append(buffer[:trace-pred_str, selected_sensor].T)
                    tag.append(0)
    if batch:
        batch = np.stack(batch)
    else:
        print("Error, no violation found!!!")
    if norm[counter_index] > 0:
        print("Warning: File ends before enough instances collected. Total counts:", norm[counter_index])
    return (batch, tag)
def load_h5_tag_grid(fname):
    with h5py.File(fname,'r') as f:
        data = f["data"].value
        tag = f["tag"].value
    max_tag = np.max(tag)
    # make tag = 2 and tag =1 the same
    if max_tag > 1.5:
        tag = np.bitwise_not(tag < 1.5)
    tag = to_categorical(tag)
    [x_train, x_test] = np.array_split(data, 2, axis=0)
    [y_train, y_test] = np.array_split(tag, 2, axis=0)
    return [x_train, y_train, x_test, y_test]
def load_frozen_lstm(model_name):
    sensor_model = load_model(model_name)
    for layer in sensor_model.layers:
        layer.trainable = False
    sensor_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return sensor_model
if __name__ == "__main__":
    full_x = []
    full_y = []
    f_list = [
        "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
        "/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR"
        "/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR"
        "/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR"
        ]
    #f_list = ["F:\\Yaswan2c\\Yaswan2c.gridIR"]
    for fname in f_list:
        #fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\test.gridIR"
        (vios_data, dim) = read_violation(fname, start_line=200,trace=40)
        (norm_data, dim) = read_violation(fname, lines_to_read=25, trace=40, count=2000, reverse=True)
        vios_data = vios_data[4:,:] # striping coordinates
        norm_data = norm_data[4:,:]
        # [vios_train, vios_test] = np.array_split(vios_data, 2, axis=1)
        # [norm_train, norm_test] = np.array_split(norm_data, 2, axis=1)

        yp_len = vios_data.shape[1]
        yn_len = norm_data.shape[1]
        y = [1] * yp_len+ [0] * yn_len
        full_y += y
        full_x.append(vios_data)
        full_x.append(norm_data)

    full_x = np.hstack(full_x).T
    shuffle_index = np.random.shuffle(range(len(full_y)))
    shuffled_x = full_x[shuffle_index,:]
    shuffled_x = np.expand_dims(shuffled_x, axis=2)
    shuffled_y = full_y[shuffle_index]
    
    save_fname = "combined_lstm_training.data"
    with h5py.File(save_fname,"w") as hf:
        hf.create_dataset("x", data=shuffled_x, dtype = 'float32')
        hf.create_dataset("y", data=shuffled_y, dtype = 'float32')