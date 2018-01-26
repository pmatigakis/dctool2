from luigi import Task, LocalTarget
from luigi.util import inherits
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np

from dctool2.categories.training import TrainPipelineUsingBestParameters
from dctool2.categories.datasets import CreateDataset


@inherits(TrainPipelineUsingBestParameters)
@inherits(CreateDataset)
class CalculateLearningCurveData(Task):
    def requires(self):
        return [
            self.clone(TrainPipelineUsingBestParameters),
            self.clone(CreateDataset)
        ]

    def output(self):
        return [
            LocalTarget("{}/learning_curve/train_sizes.pickle".format(
                self.output_folder)),
            LocalTarget("{}/learning_curve/train_scores.pickle".format(
                self.output_folder)),
            LocalTarget("{}/learning_curve/test_scores.pickle".format(
                self.output_folder))
        ]

    def run(self):
        pipeline_file, [classes_file, data_file] = self.input()

        pipeline = joblib.load(pipeline_file.path)
        x = joblib.load(data_file.path)
        y = joblib.load(classes_file.path)

        cv = ShuffleSplit(
            len(x),
            n_iter=5,
            test_size=self.test_size,
            random_state=self.random_state
        )

        training_size_range = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=pipeline,
            X=x,
            y=y,
            cv=cv,
            train_sizes=training_size_range
        )

        train_sizes_file, train_scores_file, test_scores_file = self.output()

        train_sizes_file.makedirs()
        joblib.dump(train_sizes, train_sizes_file.path)
        train_scores_file.makedirs()
        joblib.dump(train_scores, train_scores_file.path)
        test_scores_file.makedirs()
        joblib.dump(test_scores, test_scores_file.path)


@inherits(CalculateLearningCurveData)
class CreateLearningCurvePlot(Task):
    def requires(self):
        return self.clone(CalculateLearningCurveData)

    def output(self):
        return LocalTarget("{}/plots/learning_curve.png".format(
            self.output_folder))

    def run(self):
        train_sizes_file, train_scores_file, test_scores_file = self.input()

        train_sizes = joblib.load(train_sizes_file.path)
        train_scores = joblib.load(train_scores_file.path)
        test_scores = joblib.load(test_scores_file.path)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Learning curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-',
                 color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-',
                 color="g", label="Cross-validation score")
        plt.legend(loc="best")

        output_file = self.output()
        output_file.makedirs()
        plt.savefig(output_file.path)
