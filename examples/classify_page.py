from argparse import ArgumentParser

from sklearn.externals import joblib

from dctool2.common import process_web_page

import requests


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("classifier")
    parser.add_argument("binarizer")
    parser.add_argument("url")

    return parser.parse_args()


def main():
    args = get_arguments()

    classifier = joblib.load(args.classifier)
    binarizer = joblib.load(args.binarizer)

    response = requests.get(args.url)

    if response.status_code != 200:
        print("Failed to download content")
        exit()

    contents = process_web_page(response.text).lower()

    result = classifier.predict_proba([contents])

    for category, probability in zip(binarizer.classes_, result[0]):
        print(category, probability)


if __name__ == "__main__":
    main()
