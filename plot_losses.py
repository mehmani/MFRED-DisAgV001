def plot_losses(history):
    from matplotlib import pyplot 
    pyplot.plot(history.history['loss'],label='Training Loss')
    pyplot.plot(history.history['val_loss'],label='Validation Loss')
    pyplot.title('Training vs Validation Loss')
    pyplot.legend()
    pyplot.show() 