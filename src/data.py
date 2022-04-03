import numpy as np
import pandas as pd


def load_file(filepath):
    """
    load a single file as a numpy array
    """
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


def load_group(filenames, prefix=''):
    """
    load a list of files into a 3D array of [samples, timesteps, features]
    """
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


def load_dataset_group(group, prefix=''):
    """
    load a dataset group, such as train or test
    """
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def load_dataset(prefix=''):
    """
    load the dataset, returns train and test X and y elements
    """
    # load all train
    X_train, y_train = load_dataset_group('train', prefix + 'HARDataset/')
    print(X_train.shape, y_train.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    y_train = y_train - 1
    testy = testy - 1
    # one hot encode y
    y_train = pd.to_categorical(y_train)
    testy = pd.to_categorical(testy)
    print(X_train.shape, y_train.shape, testX.shape, testy.shape)
    return X_train, y_train, testX, testy
