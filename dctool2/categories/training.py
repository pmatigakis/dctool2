import logging
import json

from sklearn.externals import joblib
from luigi import Task, LocalTarget
from luigi.util import inherits

from dctool2.categories.pipelines import CreatePipeline
from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.evaluation import SelectBestPipeline


logger = logging.getLogger(__name__)


@inherits(CreatePipeline)
@inherits(TrainingDataset)
@inherits(SelectBestPipeline)
class TrainPipelineUsingBestParameters(Task):
    def output(self):
        path = "{output_folder}/trained_pipeline/pipeline.pickle".format(
            output_folder=self.output_folder
        )

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(CreatePipeline),
            self.clone(TrainingDataset),
            self.clone(SelectBestPipeline)
        ]

    def run(self):
        logger.info("training pipeline")

        (pipeline_file,
         (classes_file, data_file),
         best_pipeline_file) = self.input()

        with best_pipeline_file.open("r") as f:
            best_pipeline_report = json.loads(f.read())

        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)
        pipeline = joblib.load(pipeline_file.path)

        params = {
            "feature_extractor__max_df":
                best_pipeline_report["parameters"]["max_df"],
            "feature_extractor__min_df":
                best_pipeline_report["parameters"]["min_df"]
        }
        pipeline.set_params(**params)
        pipeline.fit(data, classes)

        trained_pipeline_file = self.output()
        trained_pipeline_file.makedirs()
        joblib.dump(pipeline, trained_pipeline_file.path)
