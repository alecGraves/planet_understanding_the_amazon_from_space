'''
This module has functions to assist with loading and manipulating data
    released as part of the kaggle competition
    https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/.

This code is available under the MIT liscense.

Created by shadySource.
'''

import os
import numpy as np
import PIL.Image as Image

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = ['C:\\data\\kaggle_satellite\\train.csv',
             'C:\\data\\kaggle_satellite\\train-tif']

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

# def load_tiff(idx, tif_dir_path=None):
#     '''
#     Loads and returns a numpified tiff file.
#     ### Inputs:
#     * idx: the number of the image to load: image_$(idx)
#     * tif_dir_path: path to the tif folder (optional, if DATA_PATH set)
#     ### Returns:
#     * numpy array of tif data
#     '''
#     # Get path from var if not passed in.
#     if tif_dir_path is None:
#         tif_dir_path = DATA_PATH[1]

#     # Get tiff
#     tif_path = os.path.join(tif_dir_path, 'train_'+str(idx)+'.tif')
#     tiff = np.float32(io.imread(tif_path))
#     tiff *= 0.00001525878 #(1/2**16)
#     return tiff

def load_jpeg(idx, jpeg_dir_path=None):
    if jpeg_dir_path is None:
        jpeg_dir_path = tif_dir_path = DATA_PATH[1]

    if type(idx).__name__ == 'str':
        jpeg_dir = idx
    else:
        jpeg_dir = os.path.join(jpeg_dir_path, 'train_'+str(idx)+'.tif')
    image = np.array(Image.open(jpeg_dir), dtype=np.float32)
    image -= np.array([1., 2., 3.], dtype=np.float32) # subtract means
    image *= np.float32(0.00392156862) #1/255
    return image

def calc_means():
    '''
    Calculates and prints the means for color channels in the images
    '''
    pass

def batch_gen(tags, val_idx, batch_size, tif_dir_path=None):
    '''
    generator object for keras training batches
    # Inputs
        tags: np.array of training tags
        val_idx: index of the beginning of validation data
        batch_size: size of every batch
        tif_dir_path: path of tif images (None for default)
    # Outputs
        yields batch_x, batch_y for use with keras model.fit_generator()
    '''
    while True:
        for i in range(0, val_idx, batch_size):
            batch_x = np.ndarray(shape=(batch_size, 256, 256, 3))
            for j in range(batch_size):
                batch_x[j] = load_jpeg(i+j, tif_dir_path)
            batch_y = tags[i:i+batch_size]

            yield batch_x, batch_y

def load_val(tags, val_idx, tif_dir_path=None, verbose=False):
    '''
    loads validation images and labels given tags and val_idx.
    # Inputs
        tags: np.array of all tags
        val_idx: index of the beginning of validation data
        tif_dir_path: path of tif images (None for default)
        verbose: print statement toggle
    # Outputs
        returns (val_x, val_y)
    '''
    val_y = tags[val_idx:]
    if verbose:
        print('Loading {0} val images.'.format(val_y.shape[0]))
    val_x = np.ndarray(shape=(val_y.shape[0], 256, 256, 3))
    for j in range(val_y.shape[0]):
        val_x[j] = load_jpeg(val_idx+j, tif_dir_path)
    return val_x, val_y


if __name__ == "__main__":
    calc_means()
    print(load_tags()[1])
    print(load_jpeg(1).shape)
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
