from api import Api

api = Api()
data = api.make_prediction(['hello this is a test', 'another news article'])

print data
