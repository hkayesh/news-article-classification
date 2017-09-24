from csv import writer
from utilities import Utilities
import glob


utilities = Utilities()
classes = ['business', 'entertainment', 'politics', 'sport', 'tech']
X = []
y = []
for cls in classes:
    path = "bbc/"+cls+"/*.txt"
    files = glob.glob(path)
    max_items_per_cat = 100
    files = files[:max_items_per_cat]

    for index, fle in enumerate(files):
        with open(fle) as f, open("{}.csv".format(fle.rsplit(".", 1)[1]),"w") as out:
            text = f.read()
            X.append(text.replace('\n', ""))
    labels = [cls]*len(files)
    new_labels = [i for i in labels]
    y = y + new_labels

final_list = zip(X, y)

utilities.save_list_as_csv(final_list, "bbc-dataset-500-rows.csv")




