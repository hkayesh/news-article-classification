import unittest
import sys
sys.path.append('/home/hmayun/PycharmProjects/test-project/')

from classifier import Preprocessor, Classifier

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


class testClassifier(unittest.TestCase):

    def test_get_classifier_model(self):
        classifier = Classifier()
        X = ['A test string', 'The second test string']
        y = ['category-A', 'category-B']
        model = classifier.get_classifier_model(X, y)
        self.assertEqual(str(type(model)), "<class 'sklearn.pipeline.Pipeline'>")


    def test_train(self):
        classifier = Classifier()
        classifier.training_file = 'news-dataset.csv'
        model = classifier.train()
        self.assertEqual(str(type(model)), "<class 'sklearn.pipeline.Pipeline'>")
        with self.assertRaises(IOError):
            classifier.training_file = 'non-existent-news-dataset.csv'
            classifier.train()

    def test_predict_category(self):
        classifier = Classifier()
        y = ['A test string']
        predicted_classes = classifier.predict_category(y)
        self.assertEqual(str(type(predicted_classes)), "<type 'list'>")
        self.assertEqual(len(y), len(predicted_classes))

        y = ['A test string.', 'The second test string.']
        predicted_classes = classifier.predict_category(y)
        self.assertEqual(str(type(predicted_classes)), "<type 'list'>")
        self.assertEqual(len(y), len(predicted_classes))
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

if __name__ == '__main__':
    unittest.main()