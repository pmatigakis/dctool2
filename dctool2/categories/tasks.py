import logging

from luigi import DateParameter, WrapperTask

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline


logger = logging.getLogger(__name__)


class CreateClassifier(WrapperTask):
    date = DateParameter()

    def requires(self):
        return [
            TrainPipeline(date=self.date),
            EvaluatePipeline(date=self.date),
        ]
