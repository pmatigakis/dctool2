import json
import itertools
import logging

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget)
from luigi.util import inherits
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, precision_score, hamming_loss
)
import numpy as np

from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.pipelines import CreateMultilabelClassifier
from dctool2.categories.common import create_classifier_id


logger = logging.getLogger(__name__)


@inherits(TrainingDataset)
@inherits(CreateMultilabelClassifier)
class EvaluateMultilabelClassifier(Task):
    max_df = FloatParameter()
    min_df = IntParameter()
    percentile = IntParameter()

    def output(self):
        path = "{output_folder}/classifier_evaluations/" \
               "{classifier_id}.json".format(
                    output_folder=self.output_folder,
                    classifier_id=create_classifier_id(
                        self.max_df, self.min_df, self.percentile)
               )

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(CreateMultilabelClassifier),
            self.clone(TrainingDataset)
        ]

    def run(self):
        logger.info("evaluating multilabel classifier")

        classifier_file, (classes_file, data_file) = self.input()

        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)

        classifier = joblib.load(classifier_file.path)
        params = {
            "feature_extractor__max_df": self.max_df,
            "feature_extractor__min_df": self.min_df,
            "feature_selector__percentile": self.percentile
        }
        classifier.estimator.set_params(**params)

        f1_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        hamming_losses = []
        kf = KFold(len(classes), random_state=self.random_state, n_folds=5)
        for train_index, test_index in kf:
            data_train, data_test = data[train_index], data[test_index]
            classes_train, classes_test = \
                classes[train_index], classes[test_index]
            classifier.fit(data_train, classes_train)
            result = classifier.predict(data_test)
            f1_scores.append(f1_score(classes_test, result, average="samples"))
            accuracy_scores.append(accuracy_score(classes_test, result))
            precision_scores.append(
                precision_score(classes_test, result, average="samples"))
            recall_scores.append(
                recall_score(classes_test, result, average="samples"))
            hamming_losses.append(hamming_loss(classes_test, result))

        result = {
            "parameters": {
                "min_df": self.min_df,
                "max_df": self.max_df,
                "percentile": self.percentile
            },
            "f1_score": np.mean(f1_scores),
            "precision_score": np.mean(precision_scores),
            "recall_score": np.mean(recall_scores),
            "accuracy_score": np.mean(accuracy_scores),
            "hamming_loss": np.mean(hamming_losses)
        }

        with self.output().open("w") as f:
            f.write(json.dumps(result))


@inherits(TrainingDataset)
class EvaluateMultilabelClassifiers(Task):
    min_df_list = ListParameter()
    max_df_list = ListParameter()
    percentile_list = ListParameter()

    def requires(self):
        pipeline_data = itertools.product(
            self.min_df_list,
            self.max_df_list,
            self.percentile_list
        )

        tasks = [
            self.clone(
                EvaluateMultilabelClassifier,
                min_df=min_df,
                max_df=max_df,
                percentile=percentile
            )
            for min_df, max_df, percentile in pipeline_data
        ]

        return tasks

    def output(self):
        return LocalTarget(
            "{}/classifier_evaluations.json".format(self.output_folder))

    def run(self):
        logger.info("evaluating multilabel classifiers")

        classifier_score_files = self.input()

        with self.output().open("w") as f:
            reports = []
            for classifier_score_file in classifier_score_files:
                with classifier_score_file.open("r") as pf:
                    evaluation = json.loads(pf.read())
                    reports.append({
                        "parameters": evaluation["parameters"],
                        "f1_score": evaluation["f1_score"],
                        "recall_score": evaluation["recall_score"],
                        "accuracy_score": evaluation["accuracy_score"],
                        "precision_score": evaluation["precision_score"],
                        "hamming_loss": evaluation["hamming_loss"]
                    })
            f.write("{}\n".format(json.dumps(reports)))


@inherits(EvaluateMultilabelClassifiers)
class SelectBestMultilabelClassifier(Task):
    def requires(self):
        return self.clone(EvaluateMultilabelClassifiers)

    def output(self):
        return LocalTarget(
            "{}/best_multilabel_classifier.json".format(self.output_folder))

    def run(self):
        logger.info("selecting best classifier")

        with self.input().open("r") as f:
            classifier_scores = json.loads(f.read())

        best_parameters = min(
            classifier_scores,
            key=lambda item: item["hamming_loss"]
        )

        with self.output().open("w") as f:
            f.write(json.dumps(best_parameters))
