'''
training function for amazonet module
'''
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping

from amazonet.models import alecnet
from amazonet.utils.data import load_tags, load_tiff
from amazonet.utils.metrics import FScore2

MODELS = [alecnet, alecnet]

csv_path = None
tif_dir_path = None

epochs = 100
batch_size = 32

tags = load_tags(csv_path)

print('Loading val data.')
val_idx = tags.shape[0]//100*95
val_y = tags[val_idx:]
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
        print("Loading model.")
        choice = np.random.choice(len(MODELS)-1)
        model = MODELS[choice].create_model()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[FScore2])

        name = "Date" + str(datetime.now()).replace(' ', "_Time").replace(":", "-").replace(".", '-') + '.h5'
        model.save_weights(name)
        savebest = ModelCheckpoint(name, monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=False)
        stopearly = EarlyStopping(monitor='val_loss', patience=10, mode='min')

        model.fit_generator(batch_gen(),
                            steps_per_epoch=val_idx//batch_size,
                            epochs=epochs,
                            validation_data=(val_x, val_y),
                            callbacks=[savebest, stopearly])


if __name__ == "__main__":
    start_training()