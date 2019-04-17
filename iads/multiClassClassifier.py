class MultiClassClassifier():
    def __init__(self):
        print("multiClassClassifier init")
        self.classifiers = {}

    def add(self, label, classifier):
        self.classifiers[label] = classifier

    def predict(self, X, without=[]):
        max = 0
        label = None
        for k,v in self.classifiers.items():
            if k not in without:
                if v.score(X) > 0 and v.score(X) > max:
                    label = k
                    max = v.score(X)
            else:
                print("without", k)
        return label
