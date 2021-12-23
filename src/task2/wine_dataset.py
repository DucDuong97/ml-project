import errno
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import gensim
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
    for _, row in data.iterrows():
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
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=10)
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   DATA PROCESSING.                                          #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def data_preprocessing(data):
    print('----------------------------------')
    print('begin data preprocessing')
    print()
    data_size = data.shape[0]
    print(f'Data size: {data_size}')
    print(data.isna().sum())

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data[['price']])
    data['price'] = imputer.transform(data[['price']])

    data['vintage'] = data['vintage'].map(lambda x: datetime.now().year - x, na_action='ignore')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data[['vintage']])
    data['vintage'] = imputer.transform(data[['vintage']])

    std = StandardScaler()
    data[['price','vintage']] = std.fit_transform(data[['price','vintage']])

    taster_twitters = data['taster_twitter_handle'].fillna(0)
    taster_twitters = [(0 if item == 0 else 1) for item in taster_twitters]
    data['taster_twitter_handle'] = taster_twitters

    data.dropna(thresh=data.shape[1]-3, inplace=True)

    encoder = ce.TargetEncoder(cols='country')
    data['country'] = encoder.fit_transform(data['country'],data['points'])

    print('..................................')
    print('end data preprocessing')
    print()
    data_loss_per = (data_size - data.shape[0])*100 / data_size
    print('Data Loss Percent: {:.2f}%'.format(data_loss_per))
    print(data.isna().sum())
    print('----------------------------------')
    return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   DATA VECTORIZING.                                         #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def transform(data, stats):
    print('----------------------------------')
    print('begin data vectorizing')
    # TODO: 1e
    # write report for it
    vectors = {}

    ###################################################

    print('begin transform country')
    countries = data['country']
    # encoder = ce.BinaryEncoder(cols=['country'])
    # countries_bin = encoder.fit_transform(countries)
    # vectors['country'] = countries_bin.to_numpy()

    # vectors['country'] = pd.get_dummies(countries).to_numpy()
    vectors['country'] = np.reshape(data['country'].to_numpy(),(data.shape[0],1))

    print('begin transform provinces')
    province = data['province']
    # encoder = ce.BinaryEncoder(cols=['province'])
    # province_bin = encoder.fit_transform(province)
    # vectors['province'] = province_bin.to_numpy()
    vectors['province'] = pd.get_dummies(province).to_numpy()

    print('begin transform region_1')
    region_1 = data['region_1']
    encoder = ce.BinaryEncoder(cols=['region_1'])
    region_1_bin = encoder.fit_transform(region_1)
    vectors['region_1'] = region_1_bin.to_numpy()

    print('begin transform region_2')
    region_2 = data['region_2']
    encoder = ce.BinaryEncoder(cols=['region_2'])
    region_2_bin = encoder.fit_transform(region_2)
    vectors['region_2'] = region_2_bin.to_numpy()

    print('begin transform winery')
    winery = data['winery']
    encoder = ce.BinaryEncoder(cols=['winery'])
    winery_bin = encoder.fit_transform(winery)
    vectors['winery'] = winery_bin.to_numpy()

    ###################################################

    print('begin transform description')
    if Path('vectorized_descriptions.csv').is_file():
        vectorized_descriptions = pd.read_csv('vectorized_descriptions.csv')
    else:
        descriptions = data['description'].tolist()
        if not Path('doc2vec.model').is_file():
            print('train model')
            model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=20)
            train_corpus = list(read_corpus(descriptions))
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model.save('doc2vec.model')
            vectorized_descriptions = [model.infer_vector(corpus.words) for corpus in train_corpus]
        else:
            model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
            train_corpus = list(read_corpus(descriptions,True))
            vectorized_descriptions = [model.infer_vector(corpus) for corpus in train_corpus]
        pd.DataFrame(vectorized_descriptions).to_csv('vectorized_descriptions.csv')
    
    vectors['description'] = np.array(vectorized_descriptions)

    ###################################################

    print('begin transform designation')
    designation = data['designation']
    encoder = ce.BinaryEncoder(cols=['designation'])
    designation_bin = encoder.fit_transform(designation)
    vectors['designation'] = designation_bin.to_numpy()

    print('begin transform variety')
    variety = data['variety']
    encoder = ce.BinaryEncoder(cols=['variety'])
    variety_bin = encoder.fit_transform(variety)
    vectors['variety'] = variety_bin.to_numpy()

    ###################################################

    print('begin transform taster_name')
    taster_names = data['taster_name']
    vectors['taster_name'] = pd.get_dummies(taster_names).to_numpy()

    print('begin transform taster_twitter_handle')
    vectors['taster_twitter_handle'] = np.reshape(data['taster_twitter_handle'].to_numpy(),(data.shape[0],1))

    ###################################################

    print('begin transform price')
    vectors['price'] = np.reshape(data['price'].to_numpy(),(data.shape[0],1))

    print('begin transform vintage')
    vectors['vintage'] = np.reshape(data['vintage'].to_numpy(),(data.shape[0],1))

    ###################################################

    # 'points': integer
    vectors['points'] = data['points'].to_numpy()

    print('end data vectorizing')
    print('----------------------------------')
    return vectors


def read_corpus(corpuses, tokens_only=False):
    for i,corpus in enumerate(corpuses):
        # print(corpus)
        corpus = gensim.parsing.preprocessing.remove_stopwords(corpus)
        # print(corpus)
        tokens = gensim.utils.simple_preprocess(corpus)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def vectorized_data():
    data = get_wine_reviews_data()
    data = extract_wine_vintage(data)
    data = data_preprocessing(data)
    stats = compute_statistics(data)
    return transform(data, stats)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   MAIN.                                                     #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
    data = extract_wine_vintage(data)
    # data = data_preprocessing(data)

    print(data.isna().sum())
    # print(data.groupby(['province','country']).count())
    # data.boxplot('points','designation')
    # plt.show()
    # print(data[data["country"].isnull()][['province','region_1','region_2','designation']])

    # transform(data,{})
    # print(compute_statistics(data))
    plot_histograms(data, {"country": 20,
                           "designation": 20,
                           "price": 80,
                           "points": 10,
                           "province":20,
                           "region_1":20,
                           "region_2": 20,
                           "taster_name":20,
                           "taster_twitter_handle": 20,
                           "variety": 20,
                           "winery": 20,
                           "vintage": 10
                           })
