import logging
import json
import pickle

from luigi import Task, LocalTarget, Parameter
from sklearn import metrics

from dctool2.categories.datasets import SplitTrainTestDataset
from dctool2.categories.pipelines import CreatePipeline
from dctool2.categories.evaluation import SelectBestPipelineParameters


logger = logging.getLogger(__name__)


class TrainPipeline(Task):
    documents_file = Parameter()
    output_folder = Parameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            ),
            CreatePipeline(
                output_folder=self.output_folder
            ),
            SelectBestPipelineParameters(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            )
        ]

    def output(self):
        return LocalTarget("{}/pipeline.pickle".format(self.output_folder))

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
    documents_file = Parameter()
    output_folder = Parameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            ),
            TrainPipeline(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            )
        ]

    def output(self):
        return LocalTarget(
            "{}/pipeline_evaluation.txt".format(self.output_folder))

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
