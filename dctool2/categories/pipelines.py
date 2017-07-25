import pickle
import logging

from luigi import Task, LocalTarget, Parameter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


logger = logging.getLogger(__name__)


class CreatePipeline(Task):
    output_folder = Parameter()

    def output(self):
        return LocalTarget(
            "{}/untrained_pipeline.pickle".format(self.output_folder))

    def run(self):
        logger.info("creating pipeline")

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
