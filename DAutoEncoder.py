def DAutoEncoder(path_to_weights):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from keras.models import load_model
    from glob import glob
    from keras.layers import Input, Dense, Flatten, MaxPooling1D, AveragePooling1D, Convolution1D,LSTM,Reshape,Bidirectional
    from keras.layers.merge import concatenate
    from keras.models import Model,model_from_json
    import keras.callbacks
    from keras.callbacks import ModelCheckpoint
    from keras.utils import plot_model
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from LossHistory import LossHistory
    from keras.layers import Dropout
    seq_length = 512
    input_seq = Input(shape = (seq_length, 1))
    conv1 = Convolution1D(filters = 8, kernel_size = 4, padding='same',
                          kernel_initializer = 'normal', activation =  'linear')(input_seq)
    flat1 = Flatten()(conv1)
    dense1 = Dense(seq_length * 8,activation='relu')(flat1)
    dense2 = Dense(128,activation='relu')(dense1)
    drop1 = Dropout(0.1)(dense2)
    dense3 = Dense(seq_length * 8,activation='relu')(drop1)
    reshaped_layer = Reshape((seq_length,-1))(dense3)
    
    conv2 = Convolution1D(filters = 1, kernel_size = 4, padding='same',
                          kernel_initializer = 'normal', activation =  'linear')(reshaped_layer)
    flat = Flatten()(conv2)
    
    # output layer
    predictions = Dense(seq_length, activation = 'linear')(flat)   
    # create the model
    model = Model(inputs=input_seq, outputs=predictions)
    # compile the model -- define the loss and the optimizer
    model.compile(loss='mean_squared_error', optimizer='Adam')
    # record the loss history
    history = LossHistory()
    # save the weigths when the vlaidation lost decreases only
    checkpointer = ModelCheckpoint(filepath=path_to_weights, save_best_only=True, verbose =1 )
    model.summary()
    return model,history,checkpointer