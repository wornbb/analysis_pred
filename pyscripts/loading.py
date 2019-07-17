import numpy as np
import pickle
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
        prev {list} -- a collection of previous CPU cycles. Only used when a trace is needed
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
            trace = [t[vios] for t in prev]
            trace = np.array(trace)
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
        buffer = deque()
        #fill que
        for i in range(trace-1):
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                buffer.append(v_formated)
            else:
                print("this is a very unlikely situation. Why would there be a blank line in the file?")
        #for i in range(start_line, start_line + lines_to_read):
        while lines_to_read and vline:
            lines_to_read -= 1
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                vios = get_violation(v_formated, prev=buffer, mode=trace, reverse=reverse)
                buffer.append(v_formated)
                if len(buffer) == trace: 
                    # The length of buffered trace will be n-1. The "-1" was compensated by the most recent voltage point.
                    # Total length of returned trace will remain n.
                    buffer.popleft()
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
def generate_prediction_data(file, lines_to_read=0, selected_sensor=[], trace=20, pred_str=5, thres=4, ref=1, global_vio=True):
    """Generate the voltage trace only at selected sensors. Generated trace batch will be balanced with violaions and normal.

    Arguments:
        file {str} -- gridIR name

    Keyword Arguments:
        lines_to_read {int} -- [description] (default: {0})
        selected_sensor {list} -- mask for selecting sensors (default: {[]})
        trace {int} -- length of the trace (default: {20})
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

        buffer = deque()
        #fill que
        for i in range(trace + pred_str -1):
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                buffer.append(v_formated)
            else:
                print("this is a very unlikely situation. Why would there be a blank line in the middle of the file?")

        if selected_sensor == "all":
            global_vio = False
            selected_sensor = np.ones_like(v_formated, dtype=int)
        counter_index = 0
        timer_index = 1
        norm_counter = 0
        norm_timer = 0
        norm = [norm_counter, norm_timer]
        local_vios = np.array([])
        while lines_to_read:
            lines_to_read -= 1
            vline = v.readline()
            v_formated = np.fromstring(vline, sep='    ', dtype='float')
            if v_formated.size:
                vios = get_violation(v_formated, prev=buffer, mode=trace)
                if not global_vio:
                    local_vios = get_violation(v_formated[selected_sensor], prev=buffer, mode=trace)
                buffer.append(v_formated)
                if len(buffer) == trace: 
                    # The length of buffered trace will be n-1. The "-1" was compensated by the most recent voltage point.
                    # Total length of returned trace will remain n.
                    buffer.popleft()
                # logic to process the grid
                if vios.size and not local_vios.size: # violation happens globally but not locally
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str
                    data = np.array(buffer)
                    batch.append(data[:trace, selected_sensor])
                    tag.append(1)
                elif local_vios.size: # local violation
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str
                    data = np.array(buffer)
                    batch.append(data[:trace, selected_sensor])
                    tag.append(2)
                else: # normal
                    if norm[counter_index] != 0: # only update timer if there is a counter.
                        norm[timer_index] -= 1
                if norm[timer_index] % (trace + pred_str)==0 and norm[counter_index] != 0:
                    norm[counter_index] -= 1
                    data = np.array(buffer)
                    batch.append(data[:trace, selected_sensor])
                    tag.append(0)
    if batch:
        batch = np.stack(batch)
    if norm[counter_index] > 0:
        print("Warning: File ends before enough instances collected. Total counts:", norm[counter_index])
    return (batch, tag)
if __name__ == "__main__":
    
    fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"
    #fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\test.gridIR"
    (vios_data, dim) = read_violation(fname, start_line=200,trace=40)
    (norm_data, dim) = read_violation(fname, lines_to_read=25, trace=40, count=2000, reverse=True)
    vios_data = vios_data[4:,:]
    norm_data = norm_data[4:,:]
    [vios_train, vios_test] = np.array_split(vios_data, 2, axis=1)
    [norm_train, norm_test] = np.array_split(norm_data, 2, axis=1)

    yp_train = vios_train.shape[1]
    yp_test = vios_test.shape[1]
    yn_train = norm_train.shape[1]
    yn_test = norm_test.shape[1]
    print(yp_train,yn_train)
    y_train = [1] * yp_train + [0] * yn_train
    y_test = [1] * yp_test + [0] * yn_test
    x_train = np.hstack((vios_train, norm_train)).T
    x_test  = np.hstack((vios_test, norm_test)).T
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    pickle.dump( [x_train,y_train,x_test,y_test], open( "all_vios.p", "wb" ) )