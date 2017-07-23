import json
import pickle
import logging

from sklearn.cross_validation import train_test_split
from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget, DateParameter)

from dctool2.datasets.documents import CreateDocumentsFile
from dctool2.common import process_web_page


logger = logging.getLogger(__name__)


class CreateDataset(Task):
    date = DateParameter()
    categories = ListParameter()

    def output(self):
        return [
            LocalTarget("data/{}/dataset/classes.pickle".format(self.date)),
            LocalTarget("data/{}/dataset/data.pickle".format(self.date))
        ]

    def requires(self):
        return CreateDocumentsFile(self.date)

    def run(self):
        classes_file, data_file = self.output()

        classes = []
        contents = []

        with self.input().open() as input_file:
            for line in input_file:
                page = json.loads(line)
                logger.info("processing %s", page["url"])
                if page["category"] in self.categories:
                    classes.append(page["category"])
                    contents.append(process_web_page(page["content"].lower()))

        with classes_file.open("w") as f2:
            pickled_classes = pickle.dumps(classes)
            f2.write(pickled_classes)

        with data_file.open("w") as f2:
            pickled_data = pickle.dumps(contents)
            f2.write(pickled_data)


class SplitTrainTestDataset(Task):
    date = DateParameter()
    test_size = FloatParameter()
    random_state = IntParameter()

    def requires(self):
        return CreateDataset(date=self.date)

    def output(self):
        base_dir = "data/{}/train_test_data".format(self.date)

        return [
            LocalTarget("{}/train_classes.pickle".format(base_dir)),
            LocalTarget("{}/train_data.pickle".format(base_dir)),
            LocalTarget("{}/test_classes.pickle".format(base_dir)),
            LocalTarget("{}/test_data.pickle".format(base_dir))
        ]

    def run(self):
        classes_file, data_file = self.input()

        with classes_file.open() as f:
            classes = pickle.loads(f.read())

        with data_file.open() as f:
            data = pickle.loads(f.read())

        data_train, data_test, classes_train, classes_test = train_test_split(
            data,
            classes,
            test_size=self.test_size,
            random_state=self.random_state
        )

        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file) = self.output()

        with classes_train_file.open("w") as f:
            data = pickle.dumps(classes_train)
            f.write(data)

        with classes_test_file.open("w") as f:
            data = pickle.dumps(classes_test)
            f.write(data)

        with data_train_file.open("w") as f:
            data = pickle.dumps(data_train)
            f.write(data)

        with data_test_file.open("w") as f:
            data = pickle.dumps(data_test)
            f.write(data)