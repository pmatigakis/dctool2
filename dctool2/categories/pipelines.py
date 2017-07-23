import pickle

from luigi import Task, LocalTarget, DateParameter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


class CreatePipeline(Task):
    date = DateParameter()

    def output(self):
        return LocalTarget(
            "data/{}/untrained_pipeline.pickle".format(self.date))

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

        classifier = CalibratedClassifierCV(SGDClassifier())

        pipeline = Pipeline([
            ("feature_extractor", feature_extractor),
            ("feature_selector", feature_selector),
            ("classifier", classifier)
        ])

        with self.output().open("w") as f:
            data = pickle.dumps(pipeline)
            f.write(data)
