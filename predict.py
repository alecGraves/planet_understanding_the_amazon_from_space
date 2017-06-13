'''
pass in model directory.
'''
import os
import argparse
import json
import numpy as np
from keras.models import load_model

from amazonet.utils.data import TAG_DICT, load_jpeg
from amazonet.utils.metrics import competition_loss, FScore2

INV_TAG_DICT = {j : i for i, j in TAG_DICT.items()}
CUSTOM_DICT = {'competition_loss' : competition_loss,
               'FScore2' : FScore2}


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
    default=os.path.join('.', 'predictions.csv'))


def get_predictions(model_path, image_path, out_file, threshold=0.4):
    '''
    Saves predictions as numpy file.
    '''
    images = [os.path.join(image_path, imagename) for imagename in os.listdir(image_path)]
    models = [os.path.join(model_path, modelname) for modelname in os.listdir(model_path)]

    predictions = []
    for modelpath in models:
        model = load_model(modelpath, custom_objects=CUSTOM_DICT)

        predictions.append([model.predict(np.expand_dims(load_jpeg(imagepath), 0))
                            for imagepath in images])
    print(predictions)
    if out_file[-5:] == '.json': 
        predictions_dict = {models[i] : {images[j] : modelpred for j, modelpred in enumerate(modelpreds)}
                            for i, modelpreds in enumerate(predictions)}

        with open(out_file, 'w') as outfile:
            json.dump(outfile, predictions_dict)
    elif out_file[-4:] == '.csv':
        predictions_csv = [i[i.find('image_', -15):-5] + ',' + 
            ' '.join([INV_TAG_DICT[j] if prednum > threshold else None for j, prednum in enumerate(p)])
            for i, p in zip(images, predictions[0])]
        print('image_name,tags\n' + '\n'.join(predictions_csv))
        # with open(out_file, 'w') as outfile:
        #     outfile.write('image_name,tags\n' + '\n'.join(predictions_csv))
    else:
        ValueError(out_file + ' must end with .csv or .json')


if __name__ == "__main__":
    args = argparser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    image_path = os.path.expanduser(args.image_path)
    out_file = os.path.expanduser(args.out_file)

    get_predictions(model_path, image_path, out_file)
