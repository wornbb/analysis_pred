import numpy as np
import pickle

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

if __name__ == "__main__":
    
    gridIR = "C:\\Users\\Yi\\Desktop\\Yaswan2c\\Yaswan2c.gridIR"


    maxlen = 5000
    start_line = 10000

    x_train = read_volt_grid(gridIR, maxlen, start_line )
    x_test = read_volt_grid(gridIR,  maxlen, maxlen + start_line)
    with open('saved_data.pk','wb') as f:
        save = [x_train, x_test]
        pickle.dump(save, f)
        print("Loading completed")
        print(x_train.shape)
