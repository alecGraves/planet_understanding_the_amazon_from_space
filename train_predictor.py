'''
training function for amazonet module
'''
import os
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from snapshot import SnapshotCallbackBuilder

from amazonet.models import incepnet, darknet19
from amazonet.utils.data import load_tags, load_tiff
from amazonet.utils.metrics import FScore2

MODELS = [incepnet, darknet19]

csv_path = None
tif_dir_path = None
model_save_path = "D:\\models"


epochs = 200
snapshots_per_train = 10
batch_size = 32

tags = load_tags(csv_path)

val_idx = tags.shape[0]//100*95
val_y = tags[val_idx:]
print('Loading {0} val images.'.format(val_y.shape[0]))
val_x = np.ndarray(shape=(val_y.shape[0], 256, 256, 4))
for j in range(val_y.shape[0]):
    val_x[j] = load_tiff(val_idx+j, tif_dir_path)

def batch_gen():
    while True:
        for i in range(0, val_idx, batch_size):
            batch_x = np.ndarray(shape=(batch_size, 256, 256, 4))
            for j in range(batch_size):
                batch_x[j] = load_tiff(i+j, tif_dir_path)
            batch_y = tags[i:i+batch_size]

            yield batch_x, batch_y


def start_training():
    while True:
        # Randomly choose an architecture.
        choice = np.random.choice(len(MODELS))
        arch = MODELS[choice]
        print("Loading model {0}.".format(arch.name))
        model = arch.create_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[FScore2])

        name = (arch.name
        + "_Date"
        + str(datetime.now()).replace(' ', "_Time").replace(":", "-").replace(".", '-') 
        + '.h5')
        name = os.path.join(model_save_path, name)
        snapshot = SnapshotCallbackBuilder(epochs, snapshots_per_train, init_lr=0.1)

        model.fit_generator(batch_gen(),
                            steps_per_epoch=val_idx//batch_size,
                            epochs=epochs,
                            validation_data=(val_x, val_y),
                            callbacks=snapshot.get_callbacks(model_prefix=name))


if __name__ == "__main__":
    start_training()