def load_data_sequence(appliance,house):
    import numpy as np
    import pandas as pd
    X_train = np.load('pre_processed/x_train_{}_{}_seq.npy'.format(appliance,house))
    Y_train = np.load('pre_processed/y_train_{}_{}_seq.npy'.format(appliance,house))
    X_valid = np.load('pre_processed/x_valid_{}_{}_seq.npy'.format(appliance,house))
    Y_valid = np.load('pre_processed/y_valid_{}_{}_seq.npy'.format(appliance,house))
    X_test = np.load('pre_processed/x_test_{}_{}_seq.npy'.format(appliance,house))
    Y_test = np.load('pre_processed/y_test_{}_{}_seq.npy'.format(appliance,house))
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test