import json
import logging
import itertools
import pickle

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   WrapperTask, Parameter, LocalTarget)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split, cross_val_score

from dctool2.datasets.tasks import CreateDocumentsFile
from dctool2.common import process_web_page


logger = logging.getLogger(__name__)


def create_classifier():
    return CalibratedClassifierCV(SGDClassifier())


class CreateDataset(Task):
    categories = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()

    def output(self):
        return [
            LocalTarget("data/dataset/classes.pickle"),
            LocalTarget("data/dataset/data.pickle")
        ]

    def requires(self):
        return CreateDocumentsFile(
            labeled_pages=self.labeled_pages,
            namenode=self.namenode,
            namenode_port=self.namenode_port
        )

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
    categories = ListParameter()
    test_size = FloatParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()

    def requires(self):
        return CreateDataset(
            categories=self.categories,
            labeled_pages=self.labeled_pages,
            namenode=self.namenode,
            namenode_port=self.namenode_port
        )

    def output(self):
        return [
            LocalTarget("data/train_test_data/train_classes.pickle"),
            LocalTarget("data/train_test_data/train_data.pickle"),
            LocalTarget("data/train_test_data/test_classes.pickle"),
            LocalTarget("data/train_test_data/test_data.pickle")
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
            test_size=self.test_size
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


class CreatePipeline(Task):
    def output(self):
        return LocalTarget("data/untrained_pipeline.pickle")

    def run(self):
        ngram_range = (1, 2)

        feature_extractor = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            max_df=0.8,
            min_df=5
        )

        feature_selector = SelectPercentile(score_func=chi2, percentile=5)

        classifier = create_classifier()

        pipeline = Pipeline([
            ("feature_extractor", feature_extractor),
            ("feature_selector", feature_selector),
            ("classifier", classifier)
        ])

        with self.output().open("w") as f:
            data = pickle.dumps(pipeline)
            f.write(data)


class PipelineCrossValScore(Task):
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = IntParameter()
    max_df = FloatParameter()
    percentile = IntParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = FloatParameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                categories=self.categories,
                test_size=self.test_size,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port
            ),
            CreatePipeline()
        ]

    def output(self):
        task_file = "pipeline_cross_val_score__min_df-{}__max_df-{}__" \
                    "percentile-{}__alpha={}.json".format(
                                             self.min_df,
                                             self.max_df,
                                             self.percentile,
                                             self.alpha
                                        )

        scores_path = "data/pipeline_cross_val_scores/{}"

        return LocalTarget(scores_path.format(task_file))

    def run(self):
        (classes_train_file,
         data_train_file,
         classes_test_file,
         data_test_file), pipeline_file = self.input()

        with classes_train_file.open() as f:
            classes = pickle.loads(f.read())

        with data_train_file.open() as f:
            data = pickle.loads(f.read())

        with pipeline_file.open() as f:
            pipeline = pickle.loads(f.read())

        parameters = {
            "feature_extractor__min_df": self.min_df,
            "feature_extractor__max_df": self.max_df,
            "feature_selector__percentile": self.percentile,
            "classifier__base_estimator__alpha": self.alpha
            # 'classifier__alpha': (0.00001, 0.000001),
        }

        pipeline.set_params(**parameters)

        scores = cross_val_score(pipeline, data, classes)

        result = {
            "parameters": {
                "feature_extractor__min_df": self.min_df,
                "feature_extractor__max_df": self.max_df,
                "feature_selector__percentile": self.percentile,
                "classifier__base_estimator__alpha": self.alpha
            },
            "scores": scores.tolist()
        }

        with self.output().open("w") as f:
            f.write(json.dumps(result))


class EvaluatePipelines(Task):
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = ListParameter()

    def requires(self):
        pipeline_data = itertools.product(
            # (3, 5, 10, 15, 20),  # min_df
            # (0.5, 0.6, 0.7, 0.8, 0.9),  # max_df
            # (5, 10, 15, 20, 25, 30, 35, 40, 45)  # percentile
            self.min_df,  # min_df
            self.max_df,  # max_df
            self.percentile,  # percentile
            self.alpha  # alpha
        )

        tasks = [
            PipelineCrossValScore(
                categories=self.categories,
                test_size=self.test_size,
                min_df=min_df,
                max_df=max_df,
                percentile=percentile,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port,
                alpha=alpha
            )
            for min_df, max_df, percentile, alpha in pipeline_data
        ]

        return tasks

    def output(self):
        return LocalTarget("data/pipeline_evaluations.txt")

    def run(self):
        pipeline_score_files = self.input()

        with self.output().open("w") as f:
            for pipeline_score_file in pipeline_score_files:
                with pipeline_score_file.open("r") as pf:
                    score = pf.read().strip()
                    f.write("{}\n".format(score))


class SelectBestPipelineParameters(Task):
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = ListParameter()

    def requires(self):
        return EvaluatePipelines(
            categories=self.categories,
            test_size=self.test_size,
            min_df=self.min_df,
            max_df=self.max_df,
            percentile=self.percentile,
            labeled_pages=self.labeled_pages,
            namenode=self.namenode,
            namenode_port=self.namenode_port,
            alpha=self.alpha
        )

    def output(self):
        return LocalTarget("data/best_pipeline_parameters.json")

    def run(self):
        parameter_scores = []

        with self.input().open("r") as f:
            for pipeline_score_data in f:
                data = json.loads(pipeline_score_data)

                parameter_scores.append(data)

        best_parameters = max(
            parameter_scores,
            key=lambda item: sum(item["scores"]) / float(len(item["scores"]))
        )

        with self.output().open("w") as f:
            f.write(json.dumps(best_parameters))


class TrainPipeline(Task):
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = ListParameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                categories=self.categories,
                test_size=self.test_size,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port
            ),
            CreatePipeline(),
            SelectBestPipelineParameters(
                categories=self.categories,
                test_size=self.test_size,
                min_df=self.min_df,
                max_df=self.max_df,
                percentile=self.percentile,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port,
                alpha=self.alpha
            )
        ]

    def output(self):
        return LocalTarget("data/pipeline.pickle")

    def run(self):
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
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = ListParameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                categories=self.categories,
                test_size=self.test_size,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port
            ),
            TrainPipeline(
                categories=self.categories,
                test_size=self.test_size,
                min_df=self.min_df,
                max_df=self.max_df,
                percentile=self.percentile,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port,
                alpha=self.alpha
            )
        ]

    def output(self):
        return LocalTarget("data/pipeline_evaluation.txt")

    def run(self):
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


class CreateClassifier(WrapperTask):
    categories = ListParameter()
    test_size = FloatParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()
    alpha = ListParameter()

    def requires(self):
        return [
            TrainPipeline(
                categories=self.categories,
                test_size=self.test_size,
                min_df=self.min_df,
                max_df=self.max_df,
                percentile=self.percentile,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port,
                alpha=self.alpha
            ),
            EvaluatePipeline(
                categories=self.categories,
                test_size=self.test_size,
                min_df=self.min_df,
                max_df=self.max_df,
                percentile=self.percentile,
                labeled_pages=self.labeled_pages,
                namenode=self.namenode,
                namenode_port=self.namenode_port,
                alpha=self.alpha
            ),
        ]
