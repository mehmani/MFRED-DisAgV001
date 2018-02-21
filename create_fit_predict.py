def create_fit_predict(model,filename,train_data,valid_data,test_data,**kwargs):
    model,history,checkpointer = model(filename)
    model_history = model.fit(train_data[0],train_data[1],validation_data=(valid_data[0],valid_data[1]),callbacks=[checkpointer],**kwargs)
    model.load_weights(filename)
    from plot_losses import plot_losses
    plot_losses(model_history)
    preds = model.predict(test_data[0])
    from scores import scores
    return scores(preds,test_data[1])