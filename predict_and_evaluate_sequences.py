def predict_and_evaluate_sequences(model,xtest,ytest):
    ypred = model.predict(xtest)
    print(ypred.shape)
    return score_sequence_other(ytest,ypred)