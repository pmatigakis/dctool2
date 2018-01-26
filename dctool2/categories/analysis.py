from luigi import Task, LocalTarget
from luigi.util import inherits
from sklearn.externals import joblib
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
)

from dctool2.categories.training import TrainPipelineUsingBestParameters
from dctool2.categories.datasets import TestDataset, CreateLabelBinarizer


@inherits(TrainPipelineUsingBestParameters)
@inherits(TestDataset)
@inherits(CreateLabelBinarizer)
class CalculateConfusionMatrix(Task):
    def output(self):
        path = "{}/analysis/confusion_matrix.txt".format(self.output_folder)

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(TrainPipelineUsingBestParameters),
            self.clone(TestDataset),
            self.clone(CreateLabelBinarizer)
        ]

    def run(self):
        pipeline_file, (classes_file, data_file), binarizer_file = self.input()

        pipeline = joblib.load(pipeline_file.path)
        binarizer = joblib.load(binarizer_file.path)
        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)

        result = pipeline.predict(data)

        with self.output().open("w") as f:
            f.write("tn, fp, fn, tp\n")

            estimator_count = len(
                pipeline.named_steps["classifier"].estimators_)
            for i in range(estimator_count):
                y = classes[:, i]

                report = confusion_matrix(y, result[:, i]).ravel().tolist()

                f.write("{} {}\n".format(binarizer.classes_[i], report))


@inherits(TrainPipelineUsingBestParameters)
@inherits(TestDataset)
class CalculateScores(Task):
    def output(self):
        path = "{}/analysis/scores.txt".format(self.output_folder)

        return LocalTarget(path)

    def requires(self):
        return [
            self.clone(TrainPipelineUsingBestParameters),
            self.clone(TestDataset)
        ]

    def run(self):
        pipeline_file, (classes_file, data_file) = self.input()

        pipeline = joblib.load(pipeline_file.path)
        classes = joblib.load(classes_file.path)
        data = joblib.load(data_file.path)

        result = pipeline.predict(data)

        score_f1 = f1_score(classes, result, average="samples")
        score_accuracy = accuracy_score(classes, result)
        score_precision = precision_score(classes, result, average="samples")
        score_recall = recall_score(classes, result, average="samples")

        with self.output().open("w") as f:
            f.write("f1 score: {}\n".format(score_f1))
            f.write("accuracy: {}\n".format(score_accuracy))
            f.write("recall: {}\n".format(score_recall))
            f.write("precision: {}".format(score_precision))
