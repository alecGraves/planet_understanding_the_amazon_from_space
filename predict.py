'''
pass in model directory.
'''
import os
import argparse
import json
import numpy as np

from amazonet.models import darknet19, alleluna
from amazonet.utils.data import load_tiff

argparser = argparse.ArgumentParser(
    description="""Pass in model directory and image directory.
    Loads models and dumps their image predictions as a .json.""")

argparser.add_argument(
    '-m',
    '--model_path',
    help="path to directory containing trained models.",
    default="D:\\models")

argparser.add_argument(
    '-i',
    '--image_path',
    help="path to directory containing images.",
    default="C:\\data\\kaggle_satellite\\train-tif")

argparser.add_argument(
    '-o',
    '--out_file',
    help="File name to save .json file.",
    default="predictions.json")


def get_predictions(model_path, image_path):
    '''
    "returns prediction stuff."
        -shadySource
    '''
    imagenum = range(len(os.listdir(image_path)))
    modelnames = os.listdir(model_path)
    predictions = []
    for name in modelnames:
        if darknet19.name in name:
            model = darknet19.create_model()
        elif alleluna.name in name:
            model = darknet19.create_model()

        model.load_weights(os.path.join(model_path, name))
        modelpreds = []
        for i in range(imagenum):
            modelpreds.append(model.predict(load_tiff(i, image_path)))
        predictions.append(modelpreds)
    predictions = np.array(predictions)
    predictions = np.average(predictions, axis=0)
    predictions = np.round(predictions)
    np.save('predictions', predictions)
    for vec in predictions:
        for i, pred in enumerate(vec):
            if 
    return 0

if __name__ == "__main__":
    args = argparser.parse_args()
    predictions = get_predictions(args.model_path, args.image_path)
    # with open(args.out_file, 'w') as out_file:
    #     json.dump(predictions, out_file)
