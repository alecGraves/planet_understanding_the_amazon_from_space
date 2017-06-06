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
from amazonet.utils.data import load_tags, batch_gen, load_val
from amazonet.utils.metrics import FScore2

MODELS = [incepnet, darknet19]

csv_path = None
tif_dir_path = None
model_save_path = "D:\\models"

epochs = 150
snapshots_per_train = 10
batch_size = 32

tags = load_tags(csv_path)
val_idx = tags.shape[0]//100*95

validation_data = load_val(tags, val_idx, tif_dir_path, True)

def start_training():
    while True:
        # Randomly choose an architecture.
        choice = np.random.choice(len(MODELS))
        arch = MODELS[choice]
        print("Loading model {0}.".format(arch.name))
        model = arch.create_model()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', FScore2])

        save_name = (arch.name + "_Date" +
                str(datetime.now()).replace(' ', "_Time").replace(":", "-").replace(".", '-'))
        save_name = os.path.join(model_save_path, save_name)
        snapshot = SnapshotCallbackBuilder(epochs, snapshots_per_train, init_lr=0.02)

        model.fit_generator(batch_gen(tags, val_idx, batch_size, tif_dir_path),
                            steps_per_epoch=val_idx//batch_size,
                            epochs=epochs,
                            validation_data=validation_data,
                            callbacks=snapshot.get_callbacks(model_prefix=save_name))


if __name__ == "__main__":
    start_training()