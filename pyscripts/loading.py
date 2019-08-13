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
def get_violation(data, occurrence=np.array([],dtype=np.double), ref=1, thres=4, prev=[], mode=0, reverse=False, return_mask=False)->np.ndarray:
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
    vios_record = []
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
        vios_record.append(vios)
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
    if return_mask:
        return [occurrence, vios_record]
    else:
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

            lines_to_read -= 1
            vline = v.readline()
            if vline == '':
                break
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
def generate_prediction_data(file, lines_to_read=0, selected_sensor=[], 
                            trace=39, pred_str=5, thres=4, ref=1, global_vio=True, balance=0.5,
                            grid_trigger=True, lstm_trigger=True):
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
    grid_batch = []
    lstm_batch = []
    grid_tag = []
    lstm_tag = []
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
                    [local_vios, vio_mask] = get_violation(v_formated[selected_sensor], prev=buffer, mode=trace, ref=ref, thres=thres, return_mask=True)
                else:
                    [vios, vio_mask] = get_violation(v_formated, prev=buffer, mode=trace, ref=ref, thres=thres, return_mask=True)
                vio_mask = vio_mask[0] # vio_mask originally a list
                # update buffer like a queue. queue is too slow for other operation
                buffer = np.roll(buffer, -1, axis=0)
                buffer[-1,:] = v_formated
                # logic to process the grid
                if vios.size and not local_vios.size: # violation happens globally but not locally
                    # update counter
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str - 1
                    # register grid data
                    if grid_trigger:
                        grid_batch.append(buffer[:trace-pred_str, selected_sensor].T)
                        grid_tag.append(2)
                    # randomly select strictly non-violation location
                    # shuffle_buffer = np.copy(vio_mask)
                    # np.random.shuffle(shuffle_buffer)
                    # non_vio = np.bitwise_and(shuffle_buffer, np.bitwise_not(vio_mask))
                    if lstm_trigger:
                        non_vio = select_other_nodes(vio_mask, balance=balance)
                    # register lstm data
                        lstm_batch.append(buffer[:trace-pred_str, vio_mask].T)
                        lstm_tag += [1] * np.sum(vio_mask)
                        lstm_batch.append(buffer[:trace-pred_str, non_vio].T)
                        lstm_tag += [0] * np.sum(non_vio)

                elif local_vios.size: # local violation
                    # update counter
                    norm[counter_index] += 1
                    norm[timer_index] += trace + pred_str - 1
                    # register grid data
                    if grid_trigger:
                        grid_batch.append(buffer[:trace-pred_str, selected_sensor].T)
                        grid_tag.append(1)                    
                    # randomly select strictly non-violation location
                    if lstm_trigger:
                        non_vio = select_other_nodes(vio_mask, balance=balance)
                        # register lstm data
                        lstm_batch.append(buffer[:trace-pred_str, vio_mask].T)
                        lstm_tag += [1] * np.sum(vio_mask)
                        lstm_batch.append(buffer[:trace-pred_str, non_vio].T)
                        lstm_tag += [0] * np.sum(non_vio)
                else: # normal
                    if norm[counter_index] != 0: # only update timer if there is a counter.
                        norm[timer_index] -= 1
                if norm[timer_index] % (trace + pred_str)==0 and norm[counter_index] != 0:
                    norm[counter_index] -= 1
                    grid_batch.append(buffer[:trace-pred_str, selected_sensor].T)
                    grid_tag.append(0)
    lstm_batch = np.vstack(lstm_batch)
    if grid_batch:
        grid_batch = np.stack(grid_batch)
    else:
        print("Error, no violation found!!!")
    if norm[counter_index] > 0:
        print("Warning: File ends before enough instances collected. Total counts:", norm[counter_index])
    return (lstm_batch, lstm_tag, grid_batch, grid_tag)


class voltnet_training_data_factory():
    def __init__(self, load_flist, lines_to_read=0, 
                            trace=39, pred_str=5, thres=4, ref=1, global_vio=True, pos_percent=0.5,
                            grid_trigger=True, grid_fsave="",  lstm_trigger=True, lstm_fsave=""):
        # sanity check
        if grid_trigger and grid_fsave == "":
            raise RuntimeError("To save grid data, a file name has to be provided")
        if lstm_trigger and lstm_fsave == "":
            raise RuntimeError("To save lstm data, a file name has to be provided")
        # saving init parameters
        self.load_flist = load_flist
        if not lines_to_read:
            self.lines_to_read = np.inf
        else:
            self.lines_to_read = lines_to_read
        self.trace = trace
        self.pred_str = pred_str
        self.thres = thres
        self.ref = ref
        self.global_vio = global_vio
        self.balance = pos_percent
        self.grid_trigger = grid_trigger
        self.grid_fsave = grid_fsave
        self.lstm_trigger = lstm_trigger
        self.lstm_fsave = lstm_fsave
        # init file structure
        [self.lstm, self.grid] = self.open_for_write()
        # extra configuration
        self.lstm_count = 0
        self.grid_count = 0
    def generate(self):
        for fname in self.load_flist:
            gridIR = self.open_for_read(fname)
            self.process_gridIR(gridIR)
            gridIR.close()
        self.lstm.close()
        self.grid.close()
    def process_gridIR(self, lf):
        buffer = self.init_buffer(lf)
        norm = {"counter": 0, "timer": 0}
        vios = np.array([])
        while self.lines_to_read > 0:
            self.lines_to_read -= 1
            line = self.read_line(lf)
            if not line.size: # eof
                print("eof")
                break
            [vios, vio_mask] = get_violation(line, prev=buffer, mode=self.trace, ref=self.ref, thres=self.thres, return_mask=True)
            vio_mask = vio_mask[0] # vio_mask originally a list
            buffer = self.update_buffer(buffer, line)
            if vios.size:
                # update norm counter
                norm["counter"] += 1
                norm["timer"] += self.trace + self.pred_str - 1
                self.register_grid(buffer, tag=1)
                self.register_lstm(vio_mask, buffer)
            else:
                if norm["counter"] != 0: # only update timer if there is a counter.
                    norm["timer"] -= 1
            if norm["timer"] % (self.trace + self.pred_str)==0 and norm["counter"] != 0:
                norm["counter"] -= 1
                self.register_grid(buffer, tag=0)
                # register random lstm for normal grid
                random_vio_percent = 0.01
                random_vio_count = int(self.grid_size * random_vio_percent)
                random_vios = np.array([1] * random_vio_count + [0] * (self.grid_size - random_vio_count), dtype=bool)
                self.register_lstm(vio_mask=random_vios, buffer=buffer, register_pos=False)
            self.lstm.flush()
            self.grid.flush()
    def register_lstm(self,  vio_mask, buffer, register_pos=True, register_neg=True):
        if self.lstm_trigger:
            norm_mask = self.select_other_nodes(vio_mask)
            vio_count = np.sum(vio_mask)
            norm_count = np.sum(norm_mask)
            new_lstm_count = self.lstm_count + vio_count * int(register_pos) + norm_count * int(register_neg)
            self.lstmX.resize(new_lstm_count, axis=0)
            self.lstmY.resize(new_lstm_count, axis=0)
            if register_pos:
                self.lstmX[self.lstm_count:self.lstm_count+vio_count,:,0] = buffer[:self.trace-self.pred_str, vio_mask].T
                self.lstmY[self.lstm_count:self.lstm_count+vio_count] = 1
            if register_neg:
                self.lstmX[self.lstm_count+vio_count* int(register_pos):new_lstm_count,:,0] = buffer[:self.trace-self.pred_str, norm_mask].T
                self.lstmY[self.lstm_count+vio_count* int(register_pos):new_lstm_count] = 0
            self.lstm_count = new_lstm_count
    def register_grid(self, buffer, tag):
        if self.grid_count:
            self.grid_count += 1
            self.gridX.resize(self.grid_count, axis=0)
            self.gridY.resize(self.grid_count, axis=0)
            self.gridX[-1,:,:,0] = buffer[:self.trace-self.pred_str, :].T
            self.gridY[-1] = tag
    def select_other_nodes(self, vio_mask):
        total_positive = np.sum(vio_mask)
        total_length = np.size(vio_mask)
        multiplier = int(1 / self.balance)
        if total_positive * multiplier <= total_length:
            shuffle_buffer = [1] * total_positive * multiplier + [0] * (total_length - total_positive * multiplier)
        else:
            shuffle_buffer = [0] * total_length
        shuffle_buffer = np.array(shuffle_buffer, dtype=bool)
        np.random.shuffle(shuffle_buffer)
        norm_mask = np.bitwise_and(shuffle_buffer, np.bitwise_not(vio_mask))
        return norm_mask
    def open_for_read(self, fname):
        fload = open(fname, 'r')
        return fload
    def open_for_write(self):
        temp_read = self.open_for_read(self.load_flist[0])
        example_line = self.read_line(temp_read)
        temp_read.close()
        self.grid_size = np.size(example_line)
        if self.lstm_trigger:
            lstm = h5py.File(self.lstm_fsave, "w")
            self.lstmX = lstm.create_dataset("x", shape=(1, self.trace - self.pred_str, 1), 
                                            maxshape=(None, self.trace - self.pred_str, 1), 
                                            chunk=(1, self.trace - self.pred_str, 1))
            self.lstmY = lstm.create_dataset("y", shape=(1,), maxshape=(None,))
        else:
            lstm = None
        if self.grid_trigger:
            grid = h5py.File(self.grid_fsave, "w")
            self.gridX = grid.create_dataset("x", shape=(1, self.grid_size, self.trace - self.pred_str, 1),
                                             maxshape=(None, self.grid_size, self.trace - self.pred_str, 1),
                                             chunk=(1, self.grid_size, self.trace - self.pred_str, 1))
            self.gridY = grid.create_dataset("y", shape=(1,), maxshape=(None,))
        else:
            grid = None
        return [lstm, grid]
    def read_line(self, lf):
        line = lf.readline()
        line_formated = np.fromstring(line, sep='   ', dtype='float')
        return line_formated
    def init_buffer(self, lf):
        buffer = []
        for i in range(self.trace):
            line = self.read_line(lf)
            self.lines_to_read -= 1
            if line.size:
                buffer.append(line)
            else:
                print("this is a very unlikely situation. Why would there be a blank line in the middle of the file?")
        buffer = np.array(buffer)
        return buffer
    def update_buffer(self, buffer, line):
        buffer = np.roll(buffer, -1, axis=0)
        buffer[-1,:] = line
        return buffer
def load_h5_grid(fname):
    with h5py.File(fname,'r') as f:
        data = f["x"].value
        tag = f["y"].value
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
def select_other_nodes(selected_nodes, balance=0.5):
    total_positive = np.sum(selected_nodes)
    total_length = np.size(selected_nodes)
    multiplier = int(1 / balance)
    if total_positive * multiplier <= total_length:
        shuffle_buffer = [1] * total_positive * multiplier + [0] * (total_length - total_positive * multiplier)
    else:
        raise ValueError("multiplier: ", multiplier ," too big")
    shuffle_buffer = np.array(shuffle_buffer)
    np.random.shuffle(shuffle_buffer)
    other_nodes = np.bitwise_and(shuffle_buffer, np.bitwise_not(selected_nodes))
    return other_nodes

class regression_training_data_factory():
    def __init__(self, load_flist, lines_to_read, start_line=0):
        self.load_flist = load_flist
        self.lines_to_read = lines_to_read
        self.start_line = start_line
    def generate(self):
        for fname in self.load_flist:
            grid_batch = read_volt_grid(fname, lines_to_read=self.lines_to_read, start_line=self.start_line)
            self.x = grid_batch
            [occurrence, vios_record] = get_violation(grid_batch, return_mask=True)
            self.y = []
            for vios in vios_record:
                if vios.any():
                    self.y.append(1)
                else:
                    self.y.append(0)
        return [self.x, self.y]

            
if __name__ == "__main__":
    full_x = []
    full_y = []
    import os
    if os.name == 'nt':
        f_list = ["F:\\Yaswan2c\\Yaswan2c.gridIR"]
    else:
        f_list = [
            "/data/yi/voltVio/analysis/raw/" + "blackscholes2c" + ".gridIR",
            "/data/yi/voltVio/analysis/raw/" + "bodytrack2c" + ".gridIR",
            "/data/yi/voltVio/analysis/raw/" + "freqmine2c"+ ".gridIR",
            "/data/yi/voltVio/analysis/raw/" + "facesim2c"+ ".gridIR",
            ]
    for fname in f_list:
        #fname = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\test.gridIR"
        if os.name == 'nt':
            (vios_data, dim) = read_violation(fname, start_line=200,trace=40, lines_to_read=100000)
        else:
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
    # save_fname = "combined_lstm_training.data"
    # with h5py.File(save_fname,"w") as hf:
    #     hf.create_dataset("x", data=full_x, dtype = 'float32')
    #     hf.create_dataset("y", data=full_y, dtype = 'float32')
    full_x = np.hstack(full_x).T
    full_y = np.array(full_y)
    shuffle_index = np.arange(len(full_y))
    np.random.shuffle(shuffle_index)
    shuffled_x = full_x[shuffle_index,:]
    shuffled_x = np.expand_dims(shuffled_x, axis=2)
    shuffled_y = full_y[shuffle_index]
    
    # save_fname = "combined_lstm_training.data"
    # with h5py.File(save_fname,"w") as hf:
    #     hf.create_dataset("x", data=shuffled_x, dtype = 'float32')
    #     hf.create_dataset("y", data=shuffled_y, dtype = 'float32')
    save_fname = "lstm_test.data"
    with h5py.File(save_fname,"w") as hf:
        hf.create_dataset("x", data=shuffled_x, dtype = 'float32')
        hf.create_dataset("y", data=shuffled_y, dtype = 'float32')