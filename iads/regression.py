import numpy as np
import math

class Regression():
    """ Classe pour représenter un regresseur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Regresseur
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, dataset):
        """construit le modele de regression sur la base fournit
        """
        raise NotImplementedError("Please implement this method")

    def accuracy(self, dataset, epsilon=0.5):
        """ Permet de calculer la qualité du système
        """
        s = 0
        for i in range(dataset.size()) :
            if self.predict(dataset.getX(i)) >=  dataset.getY(i)-epsilon and self.predict(dataset.getX(i)) < dataset.getY(i)+epsilon :
                s += 1
        return (s *1.0 / dataset.size()) *100

class GradientBatch(Regression):
    def __init__(self, input_dimension, learning_rate):
        self.dimension = input_dimension
        self.e = learning_rate
        self.theta = 0
        self.tH = []
        self.cH = []

    def cost(self, X, y):
        m = len(y)
        y_hat = np.dot(X, self.theta)
        return (1/2*m)*np.sum(np.square(y_hat-y))

    def train(self, labeledSet, verbose=False):
        indice = np.arange(labeledSet.size())
        temoin = np.random.permutation(indice)
        m = len(temoin)
        for i in temoin:
            y_hat = np.dot(labeledSet.getX(i), self.theta)
            self.theta = self.theta-(1/m)*self.e*np.dot(labeledSet.getX(i).T ,y_hat-labeledSet.getY(i))
            self.tH.append(self.theta)
            self.cH.append(self.cost(labeledSet.getX(i), labeledSet.getY(i)))
        if(verbose):
            print(self.cH[-1])
            print(self.theta)

    def predict(self, x):
        pass
