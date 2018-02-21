def Requirments():
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
    from load_data.py import load_data
    from load_data_sequence.py import load_data_sequence