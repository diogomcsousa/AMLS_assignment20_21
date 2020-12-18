from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from Utils.data_analysis import DataAnalysis
import matplotlib.pyplot as plt


def learning_curve(estimator, X, Y):
    title = "Learning Curves (SVM, Poly kernel)"
    DataAnalysis.plot_learning_curve(estimator, title, X, Y, ylim=(0.7, 1.01), n_jobs=-1)
    plt.show()


class BinaryClassifier:
    def __init__(self):
        self.clf = svm.SVC()

    def fit(self, X, Y, params):
        classifier = GridSearchCV(self.clf, params, return_train_score=True, n_jobs=-1)

        classifier.fit(X, Y)

        learning_curve(classifier.best_estimator_, X, Y)

        print("Best Estimator: \n{}\n".format(classifier.best_estimator_))
        print("Best Parameters: \n{}\n".format(classifier.best_params_))
        print("Best Validation Score: \n{}\n".format(classifier.best_score_))

        return classifier.cv_results_['mean_train_score'][classifier.best_index_], classifier

    @staticmethod
    def predict(X, Y, clf):
        pred = clf.predict(X)
        return accuracy_score(Y, pred)


class MultiClassClassifier:
    def __init__(self):
        self.clf = svm.SVC(decision_function_shape='ovo')

    def fit(self, X, Y, params):
        classifier = GridSearchCV(self.clf, params, cv=5, return_train_score=True, n_jobs=-1)

        classifier.fit(X, Y)
        learning_curve(classifier.best_estimator_, X, Y)

        print("Best Estimator: \n{}\n".format(classifier.best_estimator_))
        print("Best Parameters: \n{}\n".format(classifier.best_params_))
        print("Best Validation Score: \n{}\n".format(classifier.best_score_))

        return classifier.cv_results_['mean_train_score'][classifier.best_index_], classifier

    @staticmethod
    def predict(X, Y, clf):
        pred = clf.predict(X)
        return accuracy_score(Y, pred)
