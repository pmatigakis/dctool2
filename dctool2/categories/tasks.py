import logging

from luigi import WrapperTask, Parameter

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline


logger = logging.getLogger(__name__)


class CreateClassifier(WrapperTask):
    documents_file = Parameter()
    output_folder = Parameter()

    def requires(self):
        return [
            TrainPipeline(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            ),
            EvaluatePipeline(
                documents_file=self.documents_file,
                output_folder=self.output_folder
            ),
        ]
