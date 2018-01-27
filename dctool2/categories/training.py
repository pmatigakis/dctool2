import logging
import json

from sklearn.externals import joblib
from luigi import Task, LocalTarget
from luigi.util import inherits

from dctool2.categories.pipelines import CreateMultilabelClassifier
from dctool2.categories.datasets import TrainingDataset
from dctool2.categories.evaluation import SelectBestMultilabelClassifier


logger = logging.getLogger(__name__)


@inherits(CreateMultilabelClassifier)
@inherits(TrainingDataset)
@inherits(SelectBestMultilabelClassifier)
class TrainMultilabelClassifierUsingBestParameters(Task):
    def output(self):
        path = "{output_folder}/trained_classifier/classifier.pickle".format(
            output_folder=self.output_folder
        )

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(CreateMultilabelClassifier),
            self.clone(TrainingDataset),
            self.clone(SelectBestMultilabelClassifier)
        ]

    def run(self):
        logger.info("training classifier using best parameters")

        (classifier_file,
         (classes_file, data_file),
         best_classifier_file) = self.input()

        with best_classifier_file.open("r") as f:
            best_pipeline_report = json.loads(f.read())

        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)
        classifier = joblib.load(classifier_file.path)

        pipeline_params = {
            "feature_extractor__max_df":
                best_pipeline_report["parameters"]["max_df"],
            "feature_extractor__min_df":
                best_pipeline_report["parameters"]["min_df"],
            "feature_selector__k":
                best_pipeline_report["parameters"]["k"]
        }
        classifier.estimator.set_params(**pipeline_params)
        classifier.fit(data, classes)

        trained_classifier_file = self.output()
        trained_classifier_file.makedirs()
        joblib.dump(classifier, trained_classifier_file.path)
