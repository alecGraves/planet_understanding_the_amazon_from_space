'''
Conductor network
'''
import os
import json
import argparse
import numpy as np

from keras.optimizers import SGD
from keras.models import load_model

from amazonet.models import simple_conductor
from amazonet.utils.data import load_tags, load_val
from amazonet.utils.metrics import FScore2, competition_loss
import predict as P

argparser = argparse.ArgumentParser(
    description="Train ensemble conductor network model for kaggle satellite competition")

argparser.add_argument(
    '-t',
    '--trained_conductor',
    help='Path to a trained conductor network for evaluation.',
    default=None)

argparser.add_argument(
    '-p',
    '--predictions_path',
    help='Path to prediction json.',
    default="D:\\predictions.json")

argparser.add_argument(
    '-c',
    '--csv_path',
    help="Path to CSV file if training a conductor.",
    default=None)

args = argparser.parse_args()
json_path = os.path.expanduser(args.predictions_path)
if args.csv_path is not None:
    csv_path = os.path.expanduser(args.csv_path)
else:
    csv_path = None

# Get predictions.
with open(json_path, 'r') as json_file:
    pred_dict = json.load(json_file)

def sort_by_key_int(item):
    a, b = item
    return int(''.join(filter(str.isdigit, a)))


imagenames = []
predictions = []
for i, (modelpath, preds) in  enumerate(pred_dict.items()):
    predictions.append([])
    for imagename, pred_vec in sorted(preds.items(), key=sort_by_key_int):
        if i == 0:
            imagenames.append(imagename)
        predictions[-1].append([float(category)
                                for category in pred_vec])

predictions = np.swapaxes(predictions, 0, 1) # swap axes (images, models, 17)


# Load true values (if csv_path given).
if csv_path is not None:
    tags = load_tags(csv_path)
    val_idx = int(''.join(filter(str.isdigit, imagenames[0])))
    validation_tags = tags[val_idx:]
    conductor = simple_conductor.create_model(predictions.shape[1])
    conductor.summary()
    conductor.compile(optimizer='Adam', loss=competition_loss, metrics=[FScore2])
    conductor.fit(predictions[:-1], validation_tags, batch_size=validation_tags.shape[0], epochs=10000)
    conductor.save('trained_conductor.h5')

else:
    conductor = load_model('trained_conductor.h5', custom_objects=P.CUSTOM_DICT)
    preds = conductor.predict(predictions)
    predictions_csv = [i + ',' + P.preds_to_tags(p, threshold=0.4) for i, p in zip(imagenames, preds)]
    csv = 'image_name,tags\n' + '\n'.join(predictions_csv)
    with open('final_preds.csv', 'w') as outfile:
        outfile.write(csv)

