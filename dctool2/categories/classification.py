import logging
import json
import pickle

from luigi import Task, LocalTarget, DateParameter
from sklearn import metrics

from dctool2.categories.datasets import SplitTrainTestDataset
from dctool2.categories.pipelines import CreatePipeline
from dctool2.categories.evaluation import SelectBestPipelineParameters


logger = logging.getLogger(__name__)


class TrainPipeline(Task):
    date = DateParameter()

    def requires(self):
        return [
            SplitTrainTestDataset(date=self.date),
            CreatePipeline(date=self.date),
            SelectBestPipelineParameters(date=self.date)
        ]

    def output(self):
        return LocalTarget("data/{}/pipeline.pickle".format(self.date))

    def run(self):
        logger.info("training pipeline")

        data_files, pipeline_file, pipeline_parameters_file = self.input()

        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file) = data_files

        with classes_train_file.open() as f:
            classes = pickle.loads(f.read())

        with data_train_file.open() as f:
            data = pickle.loads(f.read())

        with pipeline_file.open() as f:
            pipeline = pickle.loads(f.read())

        with pipeline_parameters_file.open() as f:
            parameters = json.loads(f.read())["parameters"]

        pipeline.set_params(**parameters)

        pipeline.fit(data, classes)

        with self.output().open("w") as f:
            data = pickle.dumps(pipeline)
            f.write(data)


class EvaluatePipeline(Task):
    date = DateParameter()

    def requires(self):
        return [
            SplitTrainTestDataset(date=self.date),
            TrainPipeline(date=self.date)
        ]

    def output(self):
        return LocalTarget("data/{}/pipeline_evaluation.txt".format(self.date))

    def run(self):
        logger.info("evaluating pipeline")

        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file), pipeline_file = self.input()

        with classes_test_file.open() as f:
            classes = pickle.loads(f.read())

        with data_test_file.open() as f:
            data = pickle.loads(f.read())

        with pipeline_file.open() as f:
            pipeline = pickle.loads(f.read())

        results = pipeline.predict(data)

        report = metrics.classification_report(classes, results)

        with self.output().open("w") as f:
            f.write(report)
