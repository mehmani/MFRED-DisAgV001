def predict_and_evaluate(model,xtest,ytest):
    ypred = model.predict(xtest)
    return scores(ypred,ytest)