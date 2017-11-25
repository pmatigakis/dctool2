dctool2 is a collection of luigi tasks that train a web page classifier.

### Installation
Create and activate a virtualenv environment

```
virtualenv --python=python2.7 virtualenv
source virtualenv/bin/activate
```

Download and install dctool

```
python setup.py install
```

### Usage
dctool2 requires some labeled web pages to be stored in a file on an hdfs folder.
Every line on that file must contain one json encoded object per web page. The
contents of the object must have the following schema.

```python
{
    "content": "the web page content",
    "category": "the web page category"
}
```

Start the luigi scheduler

```
luigid --pidfile /path/to/pid/file --logdir /path/to/logs --state-path /path/to/state/file
```

Run the luigi tasks. The `CreateClassifier` task will perform a grid search to find the
parameters that give the best classification result. 

The following parameters must be given

| variable       | description                              |
| -------------- | ---------------------------------------- |
| documents-file | the hdfs path to the training documents  |
| output-folder  | the path to store the results            |

The rest of the configuration variables are defined in the `luigi.cfg` file.

| variable        | description                              |
|-----------------|------------------------------------------|
| categories      | what categories to use in the classifier |
| test-size       | the test set size                        |
| min-df-list     | the term minimum document frequency      |
| max-df-list     | the term maximum document frequency      |
| percentile-list | what percentile of features to keep      |
| alpha-list      | the SGDClassifier alpha value            |
| namenode-host   | the hadoop namenode address              |
| namenode-port   | the hadoop namenode port                 |

Start the task with the following command 

```
luigi --module dctool2.categories.tasks CreateClassifier \
--workers 4 \
--documents-file  "/user/panagiotis/documents/documents.json" \
--output-folder ""/home/panagiotis/projects/dctool/results"
```

The trained pipeline will be in the `<output-folder>/pipeline.pickle` file. Use scikit-learns's
`sklearn.externals.joblib` module to load it.

The classifier evaluation will be stored in the `<output-folder>/pipeline_evaluation.txt` file.

Keep in mind that training can take a long time. On a laptop with an i3-3217U CPU
and 8GB of RAM it took about an hour to train a classifier using a 2000 document
dataset with several different parameters.
