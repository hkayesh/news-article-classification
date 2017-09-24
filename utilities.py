import csv
import pickle
import sys


class Utilities(object):

    def read_from_csv(self, file_path):
        csv.field_size_limit(sys.maxsize)

        data = []

        with open(file_path, 'rb') as csvfile:
            file_data = csv.reader(csvfile, delimiter=',')
            for row in file_data:

                data.append(row)

        return data

    def convert_list_to_utf8(self, data):
        converted_data = data
        if len(data)>0 and isinstance(data[0], str):
            converted_data = [segment.decode('utf-8', 'ignore') for segment in data]
        return converted_data

    # def store_list_to_file(self,file_path, data_list):
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(data_list, f)
    #
    # def get_list_from_file(self, file_path):
    #     with open(file_path, 'rb') as f:
    #         data_list = pickle.load(f)
    #
    #     return data_list
    #
    def save_list_as_csv(self, data_list, file_path):
        with open(file_path, 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(data_list)