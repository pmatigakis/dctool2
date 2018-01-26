import logging

from luigi import WrapperTask
from luigi.util import inherits

from dctool2.categories.training import TrainPipelineUsingBestParameters
from dctool2.categories.plots import CreateLearningCurvePlot
from dctool2.categories.analysis import (
    CalculateConfusionMatrix, CalculateScores
)


logger = logging.getLogger(__name__)


@inherits(TrainPipelineUsingBestParameters)
@inherits(CalculateConfusionMatrix)
@inherits(CalculateScores)
@inherits(CreateLearningCurvePlot)
class CreateClassifier(WrapperTask):
    def requires(self):
        return [
            self.clone(TrainPipelineUsingBestParameters),
            self.clone(CalculateConfusionMatrix),
            self.clone(CalculateScores),
            self.clone(CreateLearningCurvePlot)
        ]
