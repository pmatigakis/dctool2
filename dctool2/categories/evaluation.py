import json
import itertools
import pickle
import logging
import hashlib

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget, DateParameter, Parameter)
from sklearn.cross_validation import cross_val_score

from dctool2.categories.datasets import SplitTrainTestDataset
from dctool2.categories.pipelines import CreatePipeline


logger = logging.getLogger(__name__)


class CalculatePipelineCrossValScore(Task):
    date = DateParameter()
    min_df = IntParameter()
    max_df = FloatParameter()
    percentile = IntParameter()
    alpha = FloatParameter()
    random_state = IntParameter()
    documents_file = Parameter()

    def requires(self):
        return [
            SplitTrainTestDataset(
                date=self.date,
                documents_file=self.documents_file
            ),
            CreatePipeline(date=self.date)
        ]

    def output(self):
        file_id = "min_df-{min_df}__" \
                  "max_df-{max_df}__" \
                  "percentile-{percentile}__" \
                  "alpha={alpha}__" \
                  "random_state={random_state}".format(
                      min_df=self.min_df,
                      max_df=self.max_df,
                      percentile=self.percentile,
                      alpha=self.alpha,
                      random_state=self.random_state
                  )

        file_id = hashlib.sha256(file_id).hexdigest()

        task_file = "pipeline_cross_val_score__{}.json".format(file_id)

        scores_path = "data/{}/pipeline_cross_val_scores/{}"

        return LocalTarget(scores_path.format(self.date, task_file))

    def run(self):
        logger.info("calculating pipeline cross validation score")

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
            "classifier__base_estimator__alpha": self.alpha,
            "classifier__base_estimator__random_state": self.random_state
            # 'classifier__alpha': (0.00001, 0.000001),
        }

        pipeline.set_params(**parameters)

        scores = cross_val_score(pipeline, data, classes)

        result = {
            "parameters": {
                "feature_extractor__min_df": self.min_df,
                "feature_extractor__max_df": self.max_df,
                "feature_selector__percentile": self.percentile,
                "classifier__base_estimator__alpha": self.alpha,
                "classifier__base_estimator__random_state": self.random_state
            },
            "scores": scores.tolist()
        }

        with self.output().open("w") as f:
            f.write(json.dumps(result))


class EvaluatePipelines(Task):
    date = DateParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    alpha = ListParameter()
    random_state = IntParameter()
    documents_file = Parameter()

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
            CalculatePipelineCrossValScore(
                date=self.date,
                min_df=min_df,
                max_df=max_df,
                percentile=percentile,
                alpha=alpha,
                random_state=self.random_state,
                documents_file=self.documents_file
            )
            for min_df, max_df, percentile, alpha in pipeline_data
        ]

        return tasks

    def output(self):
        return LocalTarget(
            "data/{}/pipeline_evaluations.txt".format(self.date))

    def run(self):
        logger.info("evaluating pipelines")

        pipeline_score_files = self.input()

        with self.output().open("w") as f:
            for pipeline_score_file in pipeline_score_files:
                with pipeline_score_file.open("r") as pf:
                    score = pf.read().strip()
                    f.write("{}\n".format(score))


class SelectBestPipelineParameters(Task):
    date = DateParameter()
    min_df = ListParameter()
    max_df = ListParameter()
    percentile = ListParameter()
    alpha = ListParameter()
    random_state = IntParameter()
    documents_file = Parameter()

    def requires(self):
        return EvaluatePipelines(
            date=self.date,
            min_df=self.min_df,
            max_df=self.max_df,
            percentile=self.percentile,
            alpha=self.alpha,
            random_state=self.random_state,
            documents_file=self.documents_file
        )

    def output(self):
        return LocalTarget(
            "data/{}/best_pipeline_parameters.json".format(self.date))

    def run(self):
        logger.info("selecting best pipeline features")

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
