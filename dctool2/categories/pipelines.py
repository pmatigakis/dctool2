import logging

from luigi import Task, LocalTarget
from luigi.util import inherits
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from dctool2.categories.common import Dctool2TaskBase


logger = logging.getLogger(__name__)


@inherits(Dctool2TaskBase)
class CreatePipeline(Task):
    def output(self):
        path = "{output_folder}/pipelines/pipeline.pickle".format(
            output_folder=self.output_folder
        )

        return LocalTarget(path)

    def run(self):
        logger.info("creating pipeline")

        ngram_range = (1, 1)

        feature_extractor = CountVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            max_df=0.9,
            min_df=3
        )

        classifier = OneVsRestClassifier(MultinomialNB())

        pipeline = Pipeline([
            ("feature_extractor", feature_extractor),
            ("classifier", classifier)
        ])

        output_file = self.output()
        output_file.makedirs()
        joblib.dump(pipeline, output_file.path)
