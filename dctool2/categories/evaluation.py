import json
import itertools
import logging
import hashlib

from luigi import (Task, IntParameter, FloatParameter, ListParameter,
                   LocalTarget)
from luigi.util import inherits
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.metrics import f1_score

from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.pipelines import CreatePipeline


logger = logging.getLogger(__name__)


@inherits(TrainingDataset)
@inherits(CreatePipeline)
class CalculatePipelineCrossValScore(Task):
    min_df = IntParameter()
    max_df = FloatParameter()

    def requires(self):
        return [
            self.clone(TrainingDataset),
            self.clone(CreatePipeline)
        ]

    def output(self):
        file_id = "min_df-{min_df}__" \
                  "max_df-{max_df}__" \
                  "random_state={random_state}".format(
                      min_df=self.min_df,
                      max_df=self.max_df,
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
            "feature_extractor__max_df": self.max_df
        }

        pipeline.set_params(**parameters)

        # scores = cross_val_score(pipeline, data, classes, ))
        scores = []
        kf = KFold(len(classes))
        for train_index, test_index in kf:
            data_train, data_test = data[train_index], data[test_index]
            classes_train, classes_test = classes[train_index], classes[test_index]
            pipeline.fit(data_train, classes_train)
            result = pipeline.predict(data_test)
            score = f1_score(classes_test, result, average="weighted")
            scores.append(score)

        result = {
            "parameters": {
                "feature_extractor__min_df": self.min_df,
                "feature_extractor__max_df": self.max_df,
            },
            "scores": scores
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
                CalculatePipelineCrossValScore,
                min_df=min_df,
                max_df=max_df
            )
            for min_df, max_df in pipeline_data
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
