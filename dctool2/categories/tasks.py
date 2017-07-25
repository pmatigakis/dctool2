import logging

from luigi import DateParameter, WrapperTask, Parameter

from dctool2.categories.classification import TrainPipeline, EvaluatePipeline


logger = logging.getLogger(__name__)


class CreateClassifier(WrapperTask):
    date = DateParameter()
    documents_file = Parameter()
    output_folder = Parameter()

    def requires(self):
        output_folder = "{}/{}".format(self.output_folder, self.date)

        return [
            TrainPipeline(
                documents_file=self.documents_file,
                output_folder=output_folder
            ),
            EvaluatePipeline(
                documents_file=self.documents_file,
                output_folder=output_folder
            ),
        ]
