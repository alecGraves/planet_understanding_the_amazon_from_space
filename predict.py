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
    imagenames = sorted(os.listdir(image_path), key=getint)
    modelnames = os.listdir(model_path)
    imagepaths = [os.path.join(image_path, imagename) for imagename in imagenames]
    modelpaths = [os.path.join(model_path, modelname) for modelname in modelnames]

    predictions = []
    for i, modelpath in enumerate(modelpaths):
        model = load_model(modelpath, custom_objects=CUSTOM_DICT)

        predictions.append([])
        for j, imagepath in enumerate(imagepaths):
            predictions[i].append(model.predict(np.expand_dims(load_jpeg(imagepath), 0))[0])
            print(imagenames[j] + ', ' + preds_to_str(predictions[i][-1]))

    if out_file[-5:] == '.json': 
        predictions_dict = {modelnames[i] : {imagenames[j] : modelpred for j, modelpred in enumerate(modelpreds)}
                            for i, modelpreds in enumerate(predictions)}

        with open(out_file, 'w') as outfile:
            json.dump(outfile, predictions_dict)

    elif out_file[-4:] == '.csv':
        predictions_csv = [i + ',' + preds_to_str(p, threshold) for i, p in zip(imagenames, predictions[0])]
        csv = 'image_name,tags\n' + '\n'.join(predictions_csv)
        with open(out_file, 'w') as outfile:
            outfile.write(csv)

    else:
        ValueError(out_file + ' must end with .csv or .json')

def getint(name):
    basename = name[:-4]
    num = basename.split('_')[-1]
    return int(num)

def preds_to_str(pred, threshold=0.4):
    pred_tags = []
    maxweather = 0
    for i, category in enumerate(pred):
        # Exclusivity for weather labels
        # if i < 4:
        #     if category > maxweather
        #         if len(pred_tags) > 0:
        #             pred_tags.pop()
        #         pred_tags.append(str(INV_TAG_DICT[i]))
        #         maxweather = category
        if category > threshold:
            pred_tags.append(str(INV_TAG_DICT[i]))
    return ' '.join(pred_tags)


if __name__ == "__main__":
    args = argparser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    image_path = os.path.expanduser(args.image_path)
    out_file = os.path.expanduser(args.out_file)

    get_predictions(model_path, image_path, out_file)
