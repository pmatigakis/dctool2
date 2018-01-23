import logging

from luigi import Task, LocalTarget, Parameter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


logger = logging.getLogger(__name__)


class CreatePipeline(Task):
    output_folder = Parameter()

    def output(self):
        return LocalTarget(
            "{}/untrained_pipeline.pickle".format(self.output_folder))

    def run(self):
        logger.info("creating pipeline")

        ngram_range = (1, 2)

        feature_extractor = CountVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            max_df=0.8,
            min_df=5
        )

        classifier = OneVsRestClassifier(MultinomialNB())

        pipeline = Pipeline([
            ("feature_extractor", feature_extractor),
            ("classifier", classifier)
        ])

        output_file = self.output()
        output_file.makedirs()
        joblib.dump(pipeline, output_file.path)
