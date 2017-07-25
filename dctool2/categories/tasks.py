import logging

from luigi import DateParameter, WrapperTask, Parameter

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline


logger = logging.getLogger(__name__)


class CreateClassifier(WrapperTask):
    date = DateParameter()
    documents_file = Parameter()

    def requires(self):
        return [
            TrainPipeline(
                date=self.date,
                documents_file=self.documents_file
            ),
            EvaluatePipeline(
                date=self.date,
                documents_file=self.documents_file
            ),
        ]
