def create_model_4(path_to_weights):
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
    conv1_layer =  Convolution1D(filters = 32, kernel_size = 3, padding='same',
                          kernel_initializer = 'normal', activation =  'relu')
    conv1_layer_2 =  Convolution1D(filters = 32, kernel_size = 5, padding='same',
                          kernel_initializer = 'normal', activation =  'relu')
    conv1_layer_3 =  Convolution1D(filters = 32, kernel_size = 7, padding='same',
                          kernel_initializer = 'normal', activation =  'relu')
    conv1 = conv1_layer(input_seq)
    conv2 = conv1_layer_2(input_seq)
    conv3 = conv1_layer_3(input_seq)
    merged = concatenate([conv1,conv2,conv3])
    
    # flatten the weights
    flat = Flatten()(merged)
    dense1 = Dense(128,activation='relu')(flat)
    dense2 = Dense(128,activation='relu')(dense1)
    # output layer
    predictions = Dense(3, activation = 'linear')(dense2)   
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