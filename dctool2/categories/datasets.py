import json
import logging

from sklearn.cross_validation import train_test_split
from luigi import (Task, IntParameter, FloatParameter,
                   LocalTarget, ExternalTask, Parameter, WrapperTask)
from luigi.contrib.hdfs.target import HdfsTarget
from luigi.util import inherits
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from dctool2.common import process_web_page
from dctool2.categories.common import Dctool2Task


logger = logging.getLogger(__name__)


class Documents(ExternalTask):
    documents_file = Parameter()

    def output(self):
        return HdfsTarget(self.documents_file)


@inherits(Dctool2Task)
class CreateDataset(Task):
    def output(self):
        return [
            LocalTarget(
                "{}/dataset/classes.pickle".format(self.output_folder)),
            LocalTarget("{}/dataset/data.pickle".format(self.output_folder))
        ]

    def requires(self):
        return Documents(documents_file=self.documents_file)

    def run(self):
        logger.info("creating classifier dataset")

        classes_file, data_file = self.output()

        classes = []
        contents = []

        with self.input().open() as input_file:
            for line in input_file:
                page = json.loads(line)
                logger.info("processing %s", page["url"])
                if page["category"] in self.categories:
                    classes.append([page["category"]])
                    contents.append(process_web_page(page["content"].lower()))

        classes_file.makedirs()
        joblib.dump(np.array(classes), classes_file.path)

        data_file.makedirs()
        joblib.dump(np.array(contents), data_file.path)


@inherits(CreateDataset)
class CreateLabelBinarizer(Task):
    def output(self):
        path = "{}/label_binarizer/label_binarizer.pickle"
        return LocalTarget(path.format(self.output_folder))

    def requires(self):
        return self.clone(CreateDataset)

    def run(self):
        logger.info("creating binarizer for multilabel classifier")

        classes_file, data_file = self.input()
        classes = joblib.load(classes_file.path)

        binarizer = MultiLabelBinarizer()
        binarizer.fit(classes)

        binarizer_file = self.output()
        binarizer_file.makedirs()

        joblib.dump(binarizer, binarizer_file.path)


@inherits(CreateLabelBinarizer)
@inherits(CreateDataset)
class BinarizeClasses(Task):
    def output(self):
        path = "{}/binarized_classes/binarized_classes.pickle"
        return LocalTarget(path.format(self.output_folder))

    def requires(self):
        return [
            self.clone(CreateLabelBinarizer),
            self.clone(CreateDataset)
        ]

    def run(self):
        logger.info("converting target classes to binarized data")

        binarizer_file, (classes_file, data_file) = self.input()
        binarizer = joblib.load(binarizer_file.path)
        classes = joblib.load(classes_file.path)

        binarized_classes = binarizer.transform(classes)

        binalized_classes_file = self.output()
        binalized_classes_file.makedirs()
        joblib.dump(binarized_classes, binalized_classes_file.path)


@inherits(CreateDataset)
@inherits(BinarizeClasses)
class SplitTrainTestDataset(Task):
    test_size = FloatParameter()
    random_state = IntParameter()

    def requires(self):
        return [
            self.clone(CreateDataset),
            self.clone(BinarizeClasses)
        ]

    def output(self):
        base_dir = "{}/train_test_data".format(self.output_folder)

        return [
            LocalTarget("{}/train_classes.pickle".format(base_dir)),
            LocalTarget("{}/train_data.pickle".format(base_dir)),
            LocalTarget("{}/test_classes.pickle".format(base_dir)),
            LocalTarget("{}/test_data.pickle".format(base_dir))
        ]

    def run(self):
        logger.info("creating classifier train/test dataset split")

        (classes_file, data_file), binarized_classes_file = self.input()
        classes = joblib.load(binarized_classes_file.path)
        data = joblib.load(data_file.path)

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

        classes_train_file.makedirs()
        joblib.dump(classes_train, classes_train_file.path)

        classes_test_file.makedirs()
        joblib.dump(classes_test, classes_test_file.path)

        data_train_file.makedirs()
        joblib.dump(data_train, data_train_file.path)

        data_test_file.makedirs()
        joblib.dump(data_test, data_test_file.path)


@inherits(SplitTrainTestDataset)
class TrainingDataset(WrapperTask):
    def requires(self):
        return self.clone(SplitTrainTestDataset)

    def output(self):
        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file) = self.input()

        return [classes_train_file, data_train_file]


@inherits(SplitTrainTestDataset)
class TestDataset(WrapperTask):
    def requires(self):
        return self.clone(SplitTrainTestDataset)

    def output(self):
        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file) = self.input()

        return [classes_test_file, data_test_file]
