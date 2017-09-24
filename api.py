from classifier import Classifier
import json


class Api:
    def __init__(self):
        self.classifier = Classifier()

    def make_prediction(self, texts):
        if len(texts) < 1:
            raise ValueError("Invalid json string.")

        texts = json.loads(texts)

        if str(type(texts)) != "<type 'list'>":
            raise TypeError("Input must be a json array of string.")

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
