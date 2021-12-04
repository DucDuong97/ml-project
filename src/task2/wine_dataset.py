import errno
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile

import pandas as pd
import matplotlib.pyplot as plt


from make_figures import PATH, FIG_WITDH, FIG_HEIGHT, FIG_HEIGHT_FLAT, setup_matplotlib

# Change this to the path where you want to download the dataset to
DEFAULT_ROOT = '../../data/wine'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   THE REST OF THIS FILE DOES NOT NEED TO BE MANIPULATED.    #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

URL = r'http://ml.cs.uni-kl.de/download/wine-reviews.zip'
CHECKSUM = 'c7fa81ba06ed48f290463fbf3bfff4229b68fce8aa94d6a200e1e59002e9a83c'
BUFFERSIZE = 16*1024*1024


def check_exists(root):
    return os.path.isfile(os.path.join(root, 'winemag-data-130k-v2.csv'))


def download(root=DEFAULT_ROOT):
    if check_exists(root):
        return

    # download files
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print(f'Downloading "{URL}"...')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(URL) as data:
            shutil.copyfileobj(data, tmp, BUFFERSIZE)
        tmp.seek(0)

        print('Checking SHA-256 checksum of downloaded file...')
        hasher = hashlib.sha256()
        while True:
            data = tmp.read(BUFFERSIZE)
            if len(data) == 0:
                break
            hasher.update(data)
        if hasher.hexdigest() == CHECKSUM:
            print('OK')
        else:
            print('FAILURE!')
            raise RuntimeError('The SHA-256 Hash of the downloaded file is not correct!')

        print('Extracting data...')
        with zipfile.ZipFile(tmp, 'r') as zip:
            zip.extract('winemag-data-130k-v2.csv', path=root)
        print('Done!')


def get_wine_reviews_data(root=DEFAULT_ROOT):
    if not check_exists(root):
        download(root)

    file_path = os.path.join(root, 'winemag-data-130k-v2.csv')
    data = pd.read_csv(file_path, index_col=0)
    return data


def extract_wine_vintage(data):
    if (data['vintage'] is not None):
        return
    # TODO: 1b


def plot_histograms(data):
    for (colName, colData) in data.iteritems():
        fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))

        #TODO: 1c

        fig.tight_layout()
        plt.savefig(os.path.join(PATH, f'1c_histogram_of_{colName}.pdf'))
        plt.close(fig)


def compute_statistics(data):
    stats = {}
    
    #TODO: 1d
    # give each stat a name and add it to the dict
    # write report for it

    return stats


def transform(data, stats):
    #TODO: 1e
    # write report for it
    pass



if __name__ == '__main__':
    print("Dowload data")

    setup_matplotlib()

    """
    - 129971 rows

    - columns:
        'country': string
        'description': string
        'designation': string
        'points': integer
        'price': float
        'province': string
        'region_1': string
        'region_2': string
        'taster_name': string
        'taster_twitter_handle': string begin with @
        'title': string year string (string)
        'variety': string
        'winery': string
    """
    
    data = get_wine_reviews_data()
    # print(data['winery'])