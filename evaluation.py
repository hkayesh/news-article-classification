from classifier import Classifier
from utilities import Utilities
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict


class Evaluation():

    def __init__(self):
        self.classifier = Classifier()
        self.dataset = 'bbc-dataset-500-rows.csv'
        self.utilities = Utilities()

    def evaluate_classifer(self):
        data = self.utilities.read_from_csv(self.dataset)

        X = []
        y = []
        for row in data:
            X.append(row[0])
            y.append(row[1])

        X = self.utilities.convert_list_to_utf8(X)
        model = self.classifier.get_classifier_model(X, y)

        f1_scores = cross_val_score(model, X, y, scoring='f1_micro', cv=5)
        print [round(score, 3) for score in f1_scores.tolist()]
        print("F1-score: %0.4f" % (f1_scores.mean()))
        # exit()

        # random_states = [11, 22, 33, 44, 55]
        #
        # for rs in random_states:
        #     X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=rs)
        #     model = self.classifier.get_classifier_model(X_train, y_train)
        #
        #     classes = model.labels_.classes_
        #     predicted_class = model.predict(X_test)
        #
        #     y_predicted = [classes[class_id] for class_id in predicted_class]
        #
        #     print (clsr(y_test, y_predicted))


evaluation = Evaluation()
evaluation.evaluate_classifer()