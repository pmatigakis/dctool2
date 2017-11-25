import logging

from luigi import WrapperTask
from luigi.util import inherits

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline


logger = logging.getLogger(__name__)


@inherits(TrainPipeline)
@inherits(EvaluatePipeline)
class CreateClassifier(WrapperTask):
    def requires(self):
        return [
            self.clone(TrainPipeline),
            self.clone(EvaluatePipeline)
        ]
