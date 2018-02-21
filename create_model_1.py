def create_model_1(weights_path):
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
    # first convolutional layer
    conv1_layer =  Convolution1D(filters = 16, kernel_size = 3, padding='valid',
                          kernel_initializer = 'normal', activation =  'relu')
    conv1 = conv1_layer(input_seq)
    # flatten the weights
    flat = Flatten()(conv1)
    # first dense layer
    dense1 = Dense(2000, activation = 'relu')(flat)
    #Dropout
    drop1 = Dropout(0.3)(dense1)
    # second dense layer
    dense2 = Dense(512, activation = 'relu', kernel_initializer= 'normal')(drop1)
    # output layer
    predictions = Dense(3, activation = 'linear')(dense2)   
    # create the model
    model = Model(inputs=input_seq, outputs=predictions)
    # compile the model -- define the loss and the optimizer
    model.compile(loss='mean_squared_error', optimizer='Adam')
    # record the loss history
    history = LossHistory()
    # save the weigths when the vlaidation lost decreases only
    checkpointer = ModelCheckpoint(filepath=weights_path, save_best_only=True, verbose =0 )
    model.summary()
    return model,history,checkpointer
    # fit the network using the generator of mini-batches.