import logging

from luigi import Task, Parameter, LocalTarget, IntParameter, DateParameter
from snakebite.client import Client


logger = logging.getLogger(__name__)


class CreateDocumentsFile(Task):
    date = DateParameter()
    labeled_pages = Parameter()
    namenode = Parameter()
    namenode_port = IntParameter()

    def output(self):
        return LocalTarget("data/{}/documents.json".format(self.date))

    def run(self):
        logger.info("creating documents file")

        client = Client(self.namenode, self.namenode_port, use_trash=False)

        document_count = 0

        with self.output().open("w") as output_file:
            for item in client.ls([self.labeled_pages]):
                if item["file_type"] == "f":
                    document_count += 1

                    logger.info("Adding to dataset {}".format(item["path"]))

                    document = "".join(
                        [chunk for chunk in client.text([item["path"]])]
                    ).strip()

                    output_file.write("{}\n".format(document))

        logger.info("Added {} documents to dataset".format(document_count))
