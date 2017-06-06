'''
This module has functions to assist with loading and manipulating data
    released as part of the kaggle competition
    https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/.

This code is available under the MIT liscense.

Created by shadySource.
'''

import os
import numpy as np
from skimage import io

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = ['D:\\data\\kaggle_satellite\\train.csv',
             'D:\\data\\kaggle_satellite\\train-tif']

TAG_DICT = {
    'clear' : 0,
    'haze' : 1,
    'partly_cloudy' : 2,
    'cloudy' : 3,
    'primary' : 4,
    'agriculture' : 5,
    'road' : 6,
    'water' : 7,
    'cultivation' : 8,
    'habitation' : 9,
    'bare_ground' : 10,
    'selective_logging' : 11,
    'artisinal_mine' : 12,
    'blooming' : 13,
    'slash_burn' : 14,
    'blow_down' : 15,
    'conventional_mine' : 16,
}

def get_data_path():
    '''
    returns the data path.
    ## Directory Layout:
    data_dir
    * train.csv
    * train-tif/
       * train_0.tif
       * train_1.tif
       * ...
    '''
    return DATA_PATH

def load_tags(csv_path=None):
    '''
    Loads image tags from csv data.
    ### Inputs:
    * csv_path: path to csv (optional if DATA_PATH set)
    ### returns:
    * tag_array: np-array of 1/0 float32 category vectors
    '''
    # Grab path from file if not defined.
    if csv_path is None:
        csv_path = DATA_PATH[0]

    # Get tags from the csv file.
    with open(csv_path, 'r') as csv_file:
        tags = csv_file.read().split('\n')[1:-2]
    for i, tag in enumerate(tags):
        tags[i] = tag.split(',')[1].split(' ')

    # Convert tags into an array of category vectors.
    tag_array = np.zeros(shape=(len(tags), len(TAG_DICT)))
    for i, tag in enumerate(tags):
        for name in tag:
            tag_array[i][TAG_DICT[name]] = 1

    return tag_array

def load_tiff(idx, tif_dir_path=None):
    '''
    Loads and returns a numpified tiff file.
    ### Inputs:
    * idx: the number of the image to load: image_$(idx)
    * tif_dir_path: path to the tif folder (optional, if DATA_PATH set)
    ### Returns:
    * numpy array of tif data
    '''
    # Get path from var if not passed in.
    if tif_dir_path is None:
        tif_dir_path = DATA_PATH[1]

    # Get tiff
    tif_path = os.path.join(tif_dir_path, 'train_'+str(idx)+'.tif')
    tiff = np.float64(io.imread(tif_path))
    tiff *= 0.00001525878 #(1/2**16)
    return np.float32(tiff)


if __name__ == "__main__":
    print(load_tags()[1])
    print(load_tiff[1].shape)
    # tags = load_tags()
    # gmin = 1e9
    # gmax = -1e9
    # for i in range(tags.shape[0]):
    #     tiff = load_tiff(i)[:][:][2]
    #     curmin = np.amin(tiff)
    #     curmax = np.amax(tiff)
    #     if curmin < gmin:
    #         gmin = curmin
    #         print(gmin)
    #     if curmax > gmax:
    #         gmax = curmax
    #         print(gmax)
