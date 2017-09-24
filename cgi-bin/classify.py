import json

text_data = ['string 1', 'string 2']
json_data = json.dumps(text_data)

texts = json.loads(json_data)

api_path = '/home/hmayun/PycharmProjects/test-project/api.py'

from api import Api

api_cls = Api()

results = api_cls.make_prediction(texts)

print results