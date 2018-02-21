def create_fit_predict_seq(model,filename,train_data,valid_data,test_data,**kwargs):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    model,history,checkpointer = model(filename)
    model_history = model.fit(train_data[0],train_data[1],validation_data=(valid_data[0],valid_data[1]),callbacks=[checkpointer],**kwargs)
    model.load_weights(filename)
    preds = model.predict(test_data[0])
    from plot_losses import plot_losses
    plot_losses(model_history)
    from score_sequence_other import score_sequence_other
    return score_sequence_other(preds,test_data[1])
