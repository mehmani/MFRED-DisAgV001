def ConvNNLSTMSeq(path_to_weights):
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
    conv1_layer =  Convolution1D(filters = 16, kernel_size = 3, padding='same',
                          kernel_initializer = 'normal', activation =  'relu')
    conv1 = conv1_layer(input_seq)
    # flatten the weights
    flat = Flatten()(conv1)
    reshaped = Reshape(target_shape=(seq_length,16))(flat)
    bidirectional_lstm1 = Bidirectional(LSTM(128,activation='tanh',return_sequences=True))(reshaped)
    bidirectional_lstm2 = Bidirectional(LSTM(128,activation='tanh'))(bidirectional_lstm1)
    dense1 = Dense(128,activation='relu')(bidirectional_lstm2)
    drop1 = Dropout(0.2)(dense1)
    # output layer
    predictions = Dense(seq_length, activation = 'linear')(drop1)   
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