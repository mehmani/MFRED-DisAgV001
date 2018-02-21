def score_sequence_other(y_true,y_pred,plot_results=True):
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import matplotlib.pyplot as plt
    
    Y_pred_bin = y_pred.copy()
    Y_valid_bin = y_true.copy()
    np.putmask(Y_pred_bin,Y_pred_bin >= 0.1,1)
    np.putmask(Y_valid_bin,Y_valid_bin >= 0.1,1)
    np.putmask(Y_pred_bin,Y_pred_bin < 0.1,0)
    np.putmask(Y_valid_bin,Y_valid_bin < 0.1,0)
    results_arr = []
    for i,sequence in enumerate(Y_pred_bin):
        seq_accuracy = accuracy_score(Y_valid_bin[i],Y_pred_bin[i])
        s_precision,s_recall,s_f1,s_sup = precision_recall_fscore_support(Y_valid_bin[i],Y_pred_bin[i])
        results_arr.append((seq_accuracy,s_precision[0],s_recall[0],s_f1[0]))
    summarized = np.mean(results_arr,axis=0)
    results = {
            'accuracy': summarized[0],
            'f1_score': summarized[3],
            'precision': summarized[1],
            'recall_score': summarized[2]}
    if plot_results:
        print(results)
        pd_results = pd.DataFrame.from_dict(results, orient = 'index')
        pd_results = pd_results.transpose()    
        sns.barplot(data = pd_results)
    return summarized,results
