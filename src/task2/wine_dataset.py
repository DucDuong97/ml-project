import errno
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
import re

import numpy as np
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
BUFFERSIZE = 16 * 1024 * 1024


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
    if 'vintage' in data.columns:
        return
    years = []
    for index, row in data.iterrows():
        matching_num = re.findall(r'[0-9][0-9][0-9][0-9]', row['title'])
        possible_year = [int(i) for i in matching_num if 2022 >= int(i) > 1800]
        if len(possible_year) > 0:
            years.append(max(possible_year))
        else:
            years.append(None)
    data['vintage'] = years
    return data


def plot_histograms(data, columns):
    for label, content in data.iteritems():
        # fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
        if label in columns:
            values = content.tolist()
            values = [x for x in values if str(x) != 'nan']
            if isinstance(values[0], (int, float)):
                data.hist(column=label, bins=columns[label])
            else:
                if columns[label] is not None:
                    data[label].value_counts().head(columns[label]).plot.bar()
                else:
                    data[label].value_counts().plot.bar()
            plt.show()

        # fig.tight_layout()
        # plt.savefig(os.path.join(PATH, f'1c_histogram_of_{colName}.pdf'))
        # plt.close(fig)


def compute_statistics(data):
    stats = {}
    for label, content in data.iteritems():
        values = content.tolist()
        values = [x for x in values if str(x) != 'nan']
        if isinstance(values[0], (int, float)):
            stats[f'{label}_minimum'] = min(values)
            stats[f'{label}_maximum'] = max(values)
            stats[f'{label}_average'] = sum(values) / len(values)
    return stats


def transform(data, stats):
    # TODO: 1e
    # write report for it

    vectors = {}
    # 'country': string
    print('begin transform country')
    countries = data['country']
    vectors['country'] = pd.get_dummies(countries).to_numpy()

    # 'description': string

    # 'designation': string
    # print('begin transform designation')
    # designations = data['designation']
    # vectors['designation'] = pd.get_dummies(designations).to_numpy()

    # 'price': float
    vectors['price'] = np.nan_to_num(np.reshape(data['price'].to_numpy(),(data.shape[0],1)))


    # 'province': string
    print('begin transform provinces')
    provinces = data['province']
    vectors['province'] = pd.get_dummies(provinces).to_numpy()

    # 'region_1': string
    print('begin transform region_1')
    r1s = data['region_1']
    vectors['region_1'] = pd.get_dummies(r1s).to_numpy()

    # 'region_2': string
    print('begin transform region_2')
    r2s = data['region_2']
    vectors['region_2'] = pd.get_dummies(r2s).to_numpy()

    # 'taster_name': string
    print('begin transform taster_name')
    taster_names = data['taster_name']
    vectors['taster_name'] = pd.get_dummies(taster_names).to_numpy()

    # 'taster_twitter_handle': string begin with @
    print('begin transform taster_twitter_handle')
    taster_twitters = data['taster_twitter_handle']
    vectors['taster_twitter_handle'] = pd.get_dummies(taster_twitters).to_numpy()

    # 'title': string year string (string)

    # 'variety': string
    print('begin transform variety')
    varieties = data['variety']
    vectors['variety'] = pd.get_dummies(varieties).to_numpy()

    # 'winery': string
    # print('begin transform winery')
    # wineries = data['winery']
    # vectors['winery'] = pd.get_dummies(wineries).to_numpy()

    # 'points': integer
    vectors['points'] = data['points'].to_numpy()

    # 'vintage': integer
    vectors['vintage'] = np.nan_to_num(np.reshape(data['vintage'].to_numpy(),(data.shape[0],1)))

    print('end transform')
    return vectors


def vectorized_data():
    data = get_wine_reviews_data()
    data = extract_wine_vintage(data)
    stats = compute_statistics(data)
    return transform(data, stats)


if __name__ == '__main__':
    # print("Download data")

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
    #data = extract_wine_vintage(data)
    # print(data['vintage'])
    # data.hist(column='vintage', bins=10)
    # plt.show()
    # print(compute_statistics(data))
    # plot_histograms(data, {"country": 5, "price": 3, "points": 3})
