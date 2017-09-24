import unittest
import sys
import json

sys.path.append('/home/hmayun/PycharmProjects/test-project/')
from classifier import Preprocessor, Classifier
from api import Api


class TestPreprocessor(unittest.TestCase):

    def test_lemmatize(self):
        self.preprocessor = Preprocessor()

        self.assertEqual(self.preprocessor.lemmatize('agencies', 'NNS'), 'agency')
        self.assertEqual(self.preprocessor.lemmatize('news', 'NN'), 'news')

    def test_transform(self):
        texts = ["A test string", "the second string"]

        self.preprocessor = Preprocessor()
        result = self.preprocessor.transform(texts)
        self.assertEqual(result, [['test', 'string'], ['second', 'string']])


class TestClassifier(unittest.TestCase):

    def test_get_classifier_model(self):
        classifier = Classifier()

        business = "Israel looks to US for bank chiefIsrael has asked a US banker and former International Monetary Fund director to run its central bank.Stanley Fischer, vice chairman of banking giant Citigroup, has agreed to take the Bank of Israel job subject to approval from parliament and cabinet."
        entertainment = "George Michael to perform for BBCGeorge Michael is to perform live at London's Abbey Road studios as part of a BBC Radio 2 special next month."
        tech = "Doors open at biggest gadget fairThousands of technology lovers and industry experts have gathered in Las Vegas for the annual Consumer Electronics Show (CES)."
        X = [business, entertainment, tech]
        y = ['business', 'entertainment', 'tech']

        model = classifier.get_classifier_model(X, y)
        self.assertEqual(str(type(model)), "<class 'sklearn.pipeline.Pipeline'>")

    def test_train(self):
        classifier = Classifier()
        classifier.training_file = 'bbc-dataset-500-rows.csv'
        model = classifier.train()
        self.assertEqual(str(type(model)), "<class 'sklearn.pipeline.Pipeline'>")
        with self.assertRaises(IOError):
            classifier.training_file = 'non-existent-news-dataset.csv'
            classifier.train()

    def test_predict_category(self):
        classifier = Classifier()
        X = ["Israel looks to US for bank chiefIsrael has asked a US banker and former International Monetary Fund director to run its central bank.Stanley Fischer, vice chairman of banking giant Citigroup, has agreed to take the Bank of Israel job subject to approval from parliament and cabinet."]
        predicted_classes = classifier.predict_category(X)
        self.assertEqual(str(type(predicted_classes)), "<type 'list'>")
        self.assertEqual(len(X), len(predicted_classes))

        X = ["Israel looks to US for bank chiefIsrael has asked a US banker and former International Monetary Fund director to run its central bank.Stanley Fischer, vice chairman of banking giant Citigroup, has agreed to take the Bank of Israel job subject to approval from parliament and cabinet.",
             "George Michael to perform for BBCGeorge Michael is to perform live at London's Abbey Road studios as part of a BBC Radio 2 special next month."]
        predicted_classes = classifier.predict_category(X)
        self.assertEqual(str(type(predicted_classes)), "<type 'list'>")
        self.assertEqual(len(X), len(predicted_classes))
        with self.assertRaises(ValueError):
            classifier.predict_category([])

    def test_load_saved_model(self):
        classifier = Classifier()

        model_file = 'model.pickle'
        model = classifier.load_saved_model(model_file)
        self.assertEqual(str(type(model)), "<class 'sklearn.pipeline.Pipeline'>")

        model_file = 'invalid_model.pickle'
        model = classifier.load_saved_model(model_file)
        self.assertEqual(model, None)


class test_api(unittest.TestCase):

    def test_make_prediction(self):
        self.api = Api()

        business = "Israel looks to US for bank chiefIsrael has asked a US banker and former International Monetary Fund director to run its central bank.Stanley Fischer, vice chairman of banking giant Citigroup, has agreed to take the Bank of Israel job subject to approval from parliament and cabinet."
        entertainment = "George Michael to perform for BBCGeorge Michael is to perform live at London's Abbey Road studios as part of a BBC Radio 2 special next month."
        tech = "Doors open at biggest gadget fairThousands of technology lovers and industry experts have gathered in Las Vegas for the annual Consumer Electronics Show (CES)."
        articles = json.dumps([business, entertainment, tech])

        response = self.api.make_prediction(articles)
        self.assertIsNotNone(response)

        list_from_json = json.loads(response)

        self.assertEqual(len(list_from_json), 3)
        with self.assertRaises(ValueError):
            self.api.make_prediction([])

if __name__ == '__main__':
    unittest.main()