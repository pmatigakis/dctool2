import json
import itertools
import logging
import hashlib

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget)
from luigi.util import inherits
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib

from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.pipelines import CreatePipeline


logger = logging.getLogger(__name__)


@inherits(TrainingDataset)
@inherits(CreatePipeline)
class CalculatePipelineCrossValScore(Task):
    min_df = IntParameter()
    max_df = FloatParameter()
    percentile = IntParameter()
    alpha = FloatParameter()

    def requires(self):
        return [
            self.clone(TrainingDataset),
            self.clone(CreatePipeline)
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

        scores_path = "{}/pipeline_cross_val_scores/{}"

        return LocalTarget(scores_path.format(self.output_folder, task_file))

    def run(self):
        logger.info("calculating pipeline cross validation score")

        (classes_train_file, data_train_file,), pipeline_file = self.input()

        classes = joblib.load(classes_train_file.path)
        data = joblib.load(data_train_file.path)
        pipeline = joblib.load(pipeline_file.path)

        parameters = {
            "feature_extractor__min_df": self.min_df,
            "feature_extractor__max_df": self.max_df,
            "feature_selector__percentile": self.percentile,
            "classifier__alpha": self.alpha,
            "classifier__random_state": self.random_state
        }

        pipeline.set_params(**parameters)

        scores = cross_val_score(pipeline, data, classes)

        result = {
            "parameters": {
                "feature_extractor__min_df": self.min_df,
                "feature_extractor__max_df": self.max_df,
                "feature_selector__percentile": self.percentile,
                "classifier__alpha": self.alpha,
                "classifier__random_state": self.random_state
            },
            "scores": scores.tolist()
        }

        with self.output().open("w") as f:
            f.write(json.dumps(result))


@inherits(TrainingDataset)
class EvaluatePipelines(Task):
    min_df_list = ListParameter()
    max_df_list = ListParameter()
    percentile_list = ListParameter()
    alpha_list = ListParameter()

    def requires(self):
        pipeline_data = itertools.product(
            self.min_df_list,
            self.max_df_list,
            self.percentile_list,
            self.alpha_list
        )

        tasks = [
            self.clone(
                CalculatePipelineCrossValScore,
                min_df=min_df,
                max_df=max_df,
                percentile=percentile,
                alpha=alpha
            )
            for min_df, max_df, percentile, alpha in pipeline_data
        ]

        return tasks

    def output(self):
        return LocalTarget(
            "{}/pipeline_evaluations.txt".format(self.output_folder))

    def run(self):
        logger.info("evaluating pipelines")

        pipeline_score_files = self.input()

        with self.output().open("w") as f:
            for pipeline_score_file in pipeline_score_files:
                with pipeline_score_file.open("r") as pf:
                    score = pf.read().strip()
                    f.write("{}\n".format(score))


@inherits(EvaluatePipelines)
class SelectBestPipelineParameters(Task):
    def requires(self):
        return self.clone(EvaluatePipelines)

    def output(self):
        return LocalTarget(
            "{}/best_pipeline_parameters.json".format(self.output_folder))

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
