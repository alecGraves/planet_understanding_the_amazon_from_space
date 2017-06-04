'''
training function for amazonet module
'''
import numpy as np
from datetime import datetime
from amazonet.models import alecnet
from amazonet.utils.data import load_tags, load_tiff
from amazonet.utils.metrics import FScore2

models = [alecnet, alecnet]

csv_path = None
tif_dir_path = None

epochs = 100
batch_size = 32

tags = load_tags(csv_path)

print('loading val data')
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
        choice = np.random.choice(len(models)-1)
        model = models[choice].create_model()
        print("loading model")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[FScore2])

        model.fit_generator(batch_gen(),
                            steps_per_epoch=val_idx//batch_size,
                            epochs=epochs,
                            validation_data=(val_x, val_y))
        time = str(datetime.now())
        model.save_weights(time+'.h5')

if __name__ == "__main__":
    start_training()