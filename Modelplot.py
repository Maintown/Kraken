import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

class Modelplot:
    def __init__(self, history):
        self.len = len(history.history['loss'])
        self.loss = history.history['loss']
        self.accuracy = history.history['accuracy']
        self.val_loss = history.history['val_loss']
        self.val_accuracy = history.history['val_accuracy']

    def Plotloss(self):
        plt.plot(self.len, self.loss, label='Train')
        plt.plot(self.len, self.val_loss, label='Val')
        plt.title("Lossrate Plot")
        plt.xlabel("Epoch")
        plt.ylabel("Loss rate")
        plt.xticks([*range(self.len)])
        plt.show()

    def Plotacc(self):
        plt.plot(self.len, self.accuracy, label='Train')
        plt.plot(self.len, self.val_accuracy, label='Val')
        plt.title("Accuracy Plot")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy rate")
        plt.xticks([*range(self.len)])
        plt.show()