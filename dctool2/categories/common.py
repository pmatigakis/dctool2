from luigi import Task, ListParameter, Parameter


class Dctool2Task(Task):
    categories = ListParameter()
    documents_file = Parameter()
    output_folder = Parameter()
