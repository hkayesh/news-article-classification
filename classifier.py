import os
import string
import pickle
from utilities import Utilities
from preprocessing import Preprocessor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class Classifier(object):

    def __init__(self):
        self.model_file = 'model.pickle'
        self.training_file = 'bbc-dataset-500-rows.csv'
        self.utilities = Utilities()
        self.labels = LabelEncoder()
        # self.classifier_model = SGDClassifier()
        self.classifier_model = svm.SVC(kernel='sigmoid')
        # self.classifier_model = MultinomialNB()
        # self.classifier_model = RandomForestClassifier()
        # self.classifier_model = tree.DecisionTreeClassifier()

    def _build(self, classifier_model, X, y=None):
        """
        Inner build function that builds a single model.
        """

        model = Pipeline([
            ('preprocessor', Preprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 2))),
            ('classifier', classifier_model),
        ])
        model.fit(X, y)

        return model

    def get_classifier_model(self, X, y, outpath=None):
        """
        Builds a classifer for the given list of documents and targets in two
        stages: the first does a train/test split and prints a classifier report,
        the second rebuilds the model on the entire corpus and returns it for
        operationalization.
    
        X: a list or iterable of raw strings, each representing a document.
        y: a list or iterable of labels, which will be label encoded.
    
        Can specify the classifier to build with: if a class is specified then
        this will build the model with the Scikit-Learn defaults, if an instance
        is given, then it will be used directly in the build pipeline.
    
        If outpath is given, this function will write the model as a pickle.
        If verbose, this function will print out information to the command line.
        """
        # Label encode the targets
        y = self.labels.fit_transform(y)
        X = self.utilities.convert_list_to_utf8(X)
        model= self._build(self.classifier_model, X, y)
        model.labels_ = self.labels

        if outpath is not None:
            with open(outpath, 'wb') as f:
                pickle.dump(model, f)

            # print("Model written out to {}".format(outpath))

        return model

    def train(self):
        data = self.utilities.read_from_csv(self.training_file)

        X = []
        y = []
        for row in data:
            X.append(row[0])
            y.append(row[1])
        model = self.get_classifier_model(X, y, self.model_file)

        return model

    def _load_model(self):
        if os.path.exists(self.model_file):
            model = self.load_saved_model(self.model_file)
        else:
            model = self.train()

        return model

    def predict_category(self, y_test):

        if len(y_test) < 1:
            raise ValueError("'y_test' should have at least one element.")

        model = self._load_model()

        classes = model.labels_.classes_
        predicted_class = model.predict(y_test)

        output_classes = [classes[class_id] for class_id in predicted_class]

        return output_classes

    def load_saved_model(self, model_path):
        model = None
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        return model
