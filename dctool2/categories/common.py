import hashlib

from luigi import Task, ListParameter, Parameter
from luigi.util import inherits


class Dctool2TaskBase(Task):
    output_folder = Parameter()


@inherits(Dctool2TaskBase)
class Dctool2Task(Task):
    categories = ListParameter()
    documents_file = Parameter()


def create_classifier_id(max_df, min_df):
    parameter_string = "{max_df}-{min_df}".format(
        max_df=max_df,
        min_df=min_df
    )

    return hashlib.md5(parameter_string).hexdigest()
