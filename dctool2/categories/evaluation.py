import json
import itertools
import logging

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget)
from luigi.util import inherits
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.metrics import f1_score
import numpy as np

from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.pipelines import CreatePipeline
from dctool2.categories.common import create_classifier_id


logger = logging.getLogger(__name__)


@inherits(TrainingDataset)
@inherits(CreatePipeline)
class EvaluatePipeline(Task):
    max_df = FloatParameter()
    min_df = IntParameter()

    def output(self):
        path = "{output_folder}/pipeline_evaluations/" \
               "{pipeline_id}.json".format(
                    output_folder=self.output_folder,
                    pipeline_id=create_classifier_id(self.max_df, self.min_df)
               )

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(CreatePipeline),
            self.clone(TrainingDataset)
        ]

    def run(self):
        pipeline_file, (classes_file, data_file) = self.input()

        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)

        pipeline = joblib.load(pipeline_file.path)
        params = {
            "feature_extractor__max_df": self.max_df,
            "feature_extractor__min_df": self.min_df
        }
        pipeline.set_params(**params)

        scores = []
        kf = KFold(len(classes), random_state=self.random_state, n_folds=5)
        for train_index, test_index in kf:
            data_train, data_test = data[train_index], data[test_index]
            classes_train, classes_test = \
                classes[train_index], classes[test_index]
            pipeline.fit(data_train, classes_train)
            result = pipeline.predict(data_test)
            score = f1_score(classes_test, result, average="samples")
            scores.append(score)

        result = {
            "parameters": {
                "min_df": self.min_df,
                "max_df": self.max_df,
            },
            "score": np.mean(scores)
        }

        with self.output().open("w") as f:
            f.write(json.dumps(result))


@inherits(TrainingDataset)
class EvaluatePipelines(Task):
    min_df_list = ListParameter()
    max_df_list = ListParameter()

    def requires(self):
        pipeline_data = itertools.product(
            self.min_df_list,
            self.max_df_list
        )

        tasks = [
            self.clone(
                EvaluatePipeline,
                min_df=min_df,
                max_df=max_df
            )
            for min_df, max_df in pipeline_data
        ]

        return tasks

    def output(self):
        return LocalTarget(
            "{}/pipeline_evaluations.json".format(self.output_folder))

    def run(self):
        logger.info("evaluating pipelines")

        pipeline_score_files = self.input()

        with self.output().open("w") as f:
            reports = []
            for pipeline_score_file in pipeline_score_files:
                with pipeline_score_file.open("r") as pf:
                    evaluation = json.loads(pf.read())
                    reports.append({
                        "parameters": evaluation["parameters"],
                        "score": evaluation["score"]
                    })
            f.write("{}\n".format(json.dumps(reports)))


@inherits(EvaluatePipelines)
class SelectBestPipeline(Task):
    def requires(self):
        return self.clone(EvaluatePipelines)

    def output(self):
        return LocalTarget(
            "{}/best_pipeline.json".format(self.output_folder))

    def run(self):
        logger.info("selecting best pipeline")

        with self.input().open("r") as f:
            pipeline_scores = json.loads(f.read())

        best_parameters = max(
            pipeline_scores,
            key=lambda item: item["score"]
        )

        with self.output().open("w") as f:
            f.write(json.dumps(best_parameters))
