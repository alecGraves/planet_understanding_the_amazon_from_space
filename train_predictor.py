'''
training function for amazonet module
'''
import os
import argparse
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from snapshot import SnapshotCallbackBuilder
from amazonet.models import resnet
from amazonet.utils.data import load_tags, batch_gen, load_val
from amazonet.utils.metrics import FScore2, competition_loss

MODELS = [resnet]

epochs = 150
snapshots_per_train = 10
batch_size = 32


# Arguments...
argparser = argparse.ArgumentParser(
    description="Train prediction models for kaggle satellite competition")

argparser.add_argument(
    '-ip',
    '--image_path',
    help="path to folder of image data, defaults to 'C:\\data\\kaggle_satellite\\train.csv'",
    default="C:\\data\\kaggle_satellite\\train-jpg")

argparser.add_argument(
    '-m',
    '--model_path',
    help='path to save models',
    default="D:\\models")

argparser.add_argument(
    '-c',
    '--csv_path',
    help='path to csv file, defaults to "C:\\data\\kaggle_satellite\\train-csv"',
    default='C:\\data\\kaggle_satellite\\train_v2.csv')

args = argparser.parse_args()

csv_path = os.path.expanduser(args.csv_path)
jpeg_dir_path = os.path.expanduser(args.image_path)
model_save_path = os.path.expanduser(args.model_path)

tags = load_tags(csv_path)
val_idx = tags.shape[0]//10*9

validation_data = load_val(tags, val_idx, jpeg_dir_path, True)

def start_training(model_save_path):
    while True:
        # Randomly choose an architecture.
        choice = np.random.choice(len(MODELS))
        arch = MODELS[choice]
        print("Loading model {0}.".format(arch.name))
        model = arch.create_model()
        model.compile(loss=competition_loss, optimizer='sgd', metrics=['binary_accuracy', FScore2])

        save_name = (arch.name + "_Date" +
                str(datetime.now()).replace(' ', "_Time").replace(":", "-").replace(".", '-'))
        save_name = os.path.join(model_save_path, save_name)
        snapshot = SnapshotCallbackBuilder(epochs, snapshots_per_train, init_lr=0.1)

        model.fit_generator(batch_gen(tags, val_idx, batch_size, jpeg_dir_path),
                            steps_per_epoch=val_idx//batch_size,
                            epochs=epochs,
                            validation_data=validation_data,
                            callbacks=snapshot.get_callbacks(model_prefix=save_name))



if __name__ == "__main__":
    start_training(model_save_path)
