import logging

from luigi import Task, LocalTarget
from luigi.util import inherits
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, chi2

from dctool2.categories.common import Dctool2TaskBase


logger = logging.getLogger(__name__)


@inherits(Dctool2TaskBase)
class CreateMultilabelClassifier(Task):
    def output(self):
        path = "{output_folder}/classifier/classifier.pickle".format(
            output_folder=self.output_folder
        )

        return LocalTarget(path)

    def run(self):
        logger.info("creating multilabel classifier")

        ngram_range = (1, 1)

        feature_extractor = CountVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            max_df=0.9,
            min_df=3
        )

        feature_selector = SelectKBest(
            score_func=chi2,
            k=1000
        )

        pipeline = Pipeline([
            ("feature_extractor", feature_extractor),
            ("feature_selector", feature_selector),
            ("classifier", MultinomialNB())
        ])

        classifier = OneVsRestClassifier(pipeline)

        output_file = self.output()
        output_file.makedirs()
        joblib.dump(classifier, output_file.path)
