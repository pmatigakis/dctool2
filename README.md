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
dctool2 requires some labeled web pages to be stored in an hdfs folder. That
folder must contain one json file per web page. The contents of the file must
have the following schema.

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

| variable      | description                              |
| ------------- | ---------------------------------------- |
| workers       | how many luigi workers to start          |
| categories    | what categories to use in the classifier |
| labeled-pages | the hdfs path to the labeled pages       |
| test-size     | the test set size                        |
| min-df        | the term minimum document frequency      |
| max-df        | the term maximum document frequency      |
| percentile    | what percentile of features to keep      |
| alpha         | the SGDClassifier alpha value            |
| namenode      | the hadoop namenode address              |
| namenode-port | the hadoop namenode port                 |


```
luigi --module dctool2.categories.tasks CreateClassifier --workers 4 --categories '["category_1", "category_2"]' --test-size 0.2 --min-df '[3, 5, 20]' --max-df '[0.6, 0.7, 0.8, 0.9]' --percentile '[5, 10, 15, 20]' --labeled-pages "/user/panagiotis/labelled_pages" --namenode "localhost" --namenode-port 9000 --random_state 1234 --alpha '[0.00001, 0.01, 0.001]' 
```

The trained pipeline will be in the `data/pipeline.pickle` file. Use python's
`pickle` module to load it.

The classifier evaluation will be stored in the `data/pipeline_evaluation.txt` file.

Keep in mind that training can take a long time. On a laptop with an i3-3217U CPU
and 8GB of RAM it took about an hour to train a classifier using a 2000 document
dataset.
