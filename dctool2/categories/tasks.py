import logging

from luigi import WrapperTask
from luigi.util import inherits

from dctool2.categories.training import (
    TrainMultilabelClassifierUsingBestParameters
)
from dctool2.categories.analysis import (
    CalculateConfusionMatrix, CalculateScores
)


logger = logging.getLogger(__name__)


@inherits(TrainMultilabelClassifierUsingBestParameters)
@inherits(CalculateConfusionMatrix)
@inherits(CalculateScores)
class CreateClassifier(WrapperTask):
    def requires(self):
        return [
            self.clone(TrainMultilabelClassifierUsingBestParameters),
            self.clone(CalculateConfusionMatrix),
            self.clone(CalculateScores)
        ]
