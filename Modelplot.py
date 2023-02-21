import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class Modelplot:
    """
    Modelplot is used for visualize deep learning train process
    The plot is divided by 2, Trainplot and Testplot
    
    >>> mplot = Modelplot(history)
    >>> mplot.Trainplot()
    >>> mplot.Testplot()
    """
    def __init__(self, history):
        self.len = range(len(history.history['loss']))
        self.history = history
        
    def Testplot(self):
        fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
        ax[0].plot(self.len, self.history.history['val_loss'], label='loss', marker='x', color='red')
        ax[1].plot(self.len, self.history.history['val_accuracy'], label='accuracy', marker='x')
        ax[0].set_title("Test Loss")
        ax[1].set_title("Test Accuracy")
        ax[0].legend()
        ax[1].legend()
        plt.xticks(self.len);

    def Trainplot(self):
        fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True, sharex=True)
        ax[0].plot(self.len, self.history.history['loss'], label='loss', marker='x', color='red')
        ax[1].plot(self.len, self.history.history['accuracy'], label='accuracy', marker='x')
        ax[0].set_title("Train Loss")
        ax[1].set_title("Train Accuracy")
        ax[0].legend()
        ax[1].legend()
        plt.xticks(self.len);
