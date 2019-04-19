import numpy as np
import math
import matplotlib.pyplot as plt

class Regression():
    """ Classe pour représenter un regresseur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, epsilon=0.5):
        """ Constructeur de Regresseur
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.epsilon = epsilon

    def predict(self, x):
        """ rend la prediction sur x
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, dataset):
        """construit le modele de regression sur la base fournit
        """
        raise NotImplementedError("Please implement this method")

    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système
        """
        s = 0
        for i in range(dataset.size()) :
            if self.predict(dataset.getX(i)) >=  dataset.getY(i)-self.epsilon and self.predict(dataset.getX(i)) < dataset.getY(i)+self.epsilon :
                s += 1
        return (s *1.0 / dataset.size()) *100

class GradientBatch(Regression):
    def __init__(self, learning_rate, epsilon=0.5):
        super().__init__(epsilon)
        self.alpha = learning_rate
        self.theta = np.random.randn(2, 1)

    def reshape(self, X):
        return np.c_[np.ones((len(X), 1)), X]

    def cost(self, X, y):
        m = len(y)
        y_hat = np.dot(self.reshape(X), self.theta)
        return (1/2*m)*np.sum(np.square(y_hat-y))

    def train(self, labeledSet, verbose=False):
        m = labeledSet.size()
        self.tH = np.zeros((m, 2))
        self.cH = np.zeros(m)
        indice = np.arange(labeledSet.size())
        for i in indice:
            y_hat = np.dot(self.reshape(labeledSet.getX(i)), self.theta)
            self.theta = self.theta - (1/m) * self.alpha * labeledSet.getX(i).T.dot(y_hat-labeledSet.getY(i))
            self.tH[i,:] = self.theta.T
            self.cH[i] = self.cost(labeledSet.getX(i), labeledSet.getY(i))
        if(verbose):
            plt.plot(self.cH)
            plt.xlabel('Iterations')
            plt.ylabel('J(Theta)')
            plt.title('Cost of Theta throught iterations')
            plt.legend()
            plt.show()
            print(self.cH[-1])
            print(self.theta)
            plt.plot([self.tH[i][0] for i in range(len(self.tH))], label="Theta 0")
            plt.plot([self.tH[i][1] for i in range(len(self.tH))], label="Theta 1")
            plt.xlabel('Iterations')
            plt.ylabel('Theta')
            plt.title('Values of Theta throught iterations')
            plt.legend()
            plt.show()

    def predict(self, X):
        return np.amax(self.theta[0] + self.theta[1]*X) 
