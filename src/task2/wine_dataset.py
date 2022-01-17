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
        fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
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
        plt.title(f'Histogram of {label}')
        fig.tight_layout()
        plt.savefig(os.path.join(PATH, f'1c_histogram_of_{label}.pdf'))
        plt.close(fig)


def compute_statistics(data):
    print('----------------------------------')
    print('BEGIN COMPUTING STATS')
    stats = {}
    for label, content in data.iteritems():
        values = content.tolist()
        values = np.array([x for x in values if str(x) != 'nan'])
        if isinstance(values[0], (int, float)):
            print('.')
            stats[f'{label}_minimum'] = np.min(values)
            print(f'{label}_minimum = {stats[f"{label}_minimum"]}')
            stats[f'{label}_maximum'] = np.max(values)
            print(f'{label}_maximum = {stats[f"{label}_maximum"]}')
            stats[f'{label}_mean'] = np.mean(values)
            print(f'{label}_mean = {stats[f"{label}_mean"]}')
            stats[f'{label}_median'] = np.median(values)
            print(f'{label}_median = {stats[f"{label}_median"]}')
            stats[f'{label}_std'] = np.std(values)
            print(f'{label}_std = {stats[f"{label}_std"]}')
            print('.............................')
    # label = 'country'
    # content = data[[label,'points']]
    # print(content.groupby(label).std()['points'].mean())
    print('END COMPUTING STATS')
    print('----------------------------------')
    return stats

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   DATA PROCESSING.                                          #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def data_preprocessing(data):
    print('----------------------------------')
    print('BEGIN DATA PREPROCESSING')
    print(data)
    data_size = data.shape[0]
    print(f'Data size: {data_size}')
    print(data.isna().sum())
    print('.........................')

    # handle rare data
    print('REMOVING RARE CATEGORICAL DATA')
    clear_rare_data(data, 'province')
    # clear_rare_data(data, 'region_1')
    clear_rare_data(data, 'variety')
    print('.........................')

    # handle missing data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data[['price']])
    data['price'] = imputer.transform(data[['price']])

    data['vintage'] = data['vintage'].map(lambda x: datetime.now().year - x, na_action='ignore')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data[['vintage']])
    data['vintage'] = imputer.transform(data[['vintage']])

    taster_twitters = data['taster_twitter_handle'].fillna(0)
    taster_twitters = [(0 if item == 0 else 1) for item in taster_twitters]
    data['taster_twitter_handle'] = taster_twitters

    # standardize data
    std = StandardScaler()
    data[['price','vintage']] = std.fit_transform(data[['price','vintage']])

    print('END DATA PREPROCESSING')
    print()
    data_loss_per = (data_size - data.shape[0])*100 / data_size
    print('Data Loss Percent: {:.2f}%'.format(data_loss_per))
    print(data.isna().sum())
    print('----------------------------------')
    return data

def clear_rare_data(data, label, min_count=2):
    print('.')
    content = data[label]
    old_distinct_count = content.value_counts().count()
    print(f'{label} distinct count: {old_distinct_count}')
    content.mask(content.map(content.value_counts()) < min_count, 'Other',inplace=True)
    new_distinct_count = content.value_counts().count()
    print(f'Removed amount: {old_distinct_count - new_distinct_count}')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                             #
#   DATA VECTORIZING.                                         #
#                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def transform(data):
    print('----------------------------------')
    print('BEGIN DATA TRANSFORMING')
    # TODO: 1e
    # write report for it
    vectors = {}

    ###################################################

    print('begin transform country')
    vectors['country'] = sum_encode(data, 'country')

    print('begin transform provinces')
    vectors['province'] = binary_encode(data, 'province')

    print('begin transform region_1')
    vectors['region_1'] = target_encode(data, 'region_1', 'points', m=0.5)

    print('begin transform region_2')
    vectors['region_2'] = sum_encode(data, 'region_2')

    print('begin transform winery')
    vectors['winery'] = target_encode(data, 'winery', 'points', m=0.5)

    ###################################################

    print('begin transform description')
    vectors['description'] = doc2vec_encode(data,'description',vector_size=40,epochs=40)

    ###################################################

    print('begin transform designation')
    vectors['designation'] = target_encode(data,'designation','points',m=0.5)

    print('begin transform variety')
    vectors['variety'] = binary_encode(data,'variety')

    ###################################################

    print('begin transform taster_name')
    vectors['taster_name'] =  sum_encode(data, 'taster_name')

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

    print('END DATA TRANSFORMING')
    print('----------------------------------')
    return vectors


def sum_encode(data, source):
    encoder = ce.SumEncoder(cols=[source])
    vector = encoder.fit_transform(data[source])
    return vector.to_numpy()


def binary_encode(data, source):
    encoder = ce.BinaryEncoder(cols=[source])
    vector = encoder.fit_transform(data[source])
    return vector.to_numpy()


def target_encode(data, source, target, m=0):
    encoder = ce.MEstimateEncoder(cols=source,m=m)
    vector = encoder.fit_transform(data[source],data[target])
    return np.reshape(vector.to_numpy(),(data.shape[0],1))


def doc2vec_encode(data, label, vector_size=40, epochs=20):
    vec_data_path = f'./vectorized_data/vectorized_{label}_vs_{vector_size}_epo_{epochs}.csv'
    model_path = f'./doc2vec_model/doc2vec_{label}_vs_{vector_size}_epo_{epochs}.model'
    if Path(vec_data_path).is_file():
        vec_content = pd.read_csv(vec_data_path)
        vec_content = vec_content.iloc[data.index.values.tolist(),:]
        vec_content.drop(columns=vec_content.columns[0], axis=1,inplace=True)
    else:
        content = data[label].tolist()
        if not Path(model_path).is_file():
            print('train model')
            model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)
            train_corpus = list(read_corpus(content))
            model.build_vocab(train_corpus)
            model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
            model.save(model_path)
            vec_content = [model.infer_vector(corpus.words) for corpus in train_corpus]
        else:
            model = gensim.models.doc2vec.Doc2Vec.load(model_path)
            train_corpus = list(read_corpus(content,True))
            vec_content = [model.infer_vector(corpus) for corpus in train_corpus]
        pd.DataFrame(vec_content).to_csv(vec_data_path)
    return np.array(vec_content)


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
    return transform(data)

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
    # print(data['designation'].value_counts(normalize=True,ascending=True))
    # print(data.groupby(['province','country']).count())
    # data.boxplot('price')
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
                           "vintage": 80
                           })
