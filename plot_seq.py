def plot_seq(Y_pred,Y_valid_sequences):
    import matplotlib.pyplot as plt2
    import seaborn as sns
    plt2.figure(figsize=(12,4))
    sns.tsplot(Y_pred)
    sns.tsplot(Y_valid_sequences,color='red')
    plt2.legend()
    plt2.show()