from classifier import Classifier
import json

class Api():
    def __init__(self):
        self.classifier = Classifier()

    def make_prediction(self, texts):
        predictions = self.classifier.predict_category(texts)

        data = []

        for index, text in enumerate(texts):
            prediction_info = {
                'text': text,
                'category': predictions[index]
            }
            data.append(prediction_info)

        json_data = json.dumps(data)

        return json_data
