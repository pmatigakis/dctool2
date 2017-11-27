import logging

from luigi import WrapperTask
from luigi.util import inherits

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline
from dctool2.categories.plots import CreateLearningCurvePlot


logger = logging.getLogger(__name__)


@inherits(TrainPipeline)
@inherits(EvaluatePipeline)
@inherits(CreateLearningCurvePlot)
class CreateClassifier(WrapperTask):
    def requires(self):
        return [
            self.clone(TrainPipeline),
            self.clone(EvaluatePipeline),
            self.clone(CreateLearningCurvePlot)
        ]
