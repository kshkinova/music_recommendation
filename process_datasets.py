import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

def main(dataset_path, dev):
     
    save_path = './training_data/'

    train_set = pd.read_csv(dataset_path + 'train.csv', na_values=[''])

    train_set, valid_set = split_dataset(train_set)

    train_data, train_target = merge_datasets(train_set, dataset_path)
    valid_data, valid_target = merge_datasets(valid_set, dataset_path)

    if dev:
        get_plots(train_data,valid_data)

    train_data = fill_NA_values(train_data)
    valid_data = fill_NA_values(valid_data)

    train_data = extract_features(train_data)
    valid_data = extract_features(valid_data)

    train_data, valid_data = encode_features(train_data, valid_data, './encoders/')

    train_data.to_csv(save_path + 'train_data.csv', index=False)
    train_target.to_csv(save_path + 'train_target.csv', index=False)
    valid_data.to_csv(save_path + 'valid_data.csv', index=False)
    valid_target.to_csv(save_path + 'valid_target.csv', index=False)


def split_dataset(train):

    print('\ntarget = 1 (in %):', (train['target']==1).sum()/train.shape[0]*100)
    print('target = 0 (in %):', (train['target']==0).sum()/train.shape[0]*100)

    train_i = int(train.shape[0]*0.9)
    print('\nsplit index: ', train_i)

    train_set = train.iloc[:train_i, :]
    valid_set = train.iloc[train_i:, :]

    print('\nTraining set shape: ', train_set.shape)
    print('Validation set shape: ', valid_set.shape)

    return train_set, valid_set


def merge_datasets(set, path):
    '''
        Merges train/valid dataset with members, songs, and song_extra_info
    '''
    members = pd.read_csv(path + 'members.csv')
    songs = pd.read_csv(path + 'songs.csv')
    songs_extra = pd.read_csv(path + 'song_extra_info.csv')

    merged = set.merge(members, how='left', on='msno')\
        .merge(songs, how='left', on='song_id')\
        .merge(songs_extra, how='left', on='song_id')
    
    features = merged.loc[:, merged.columns != 'target']
    target = merged.loc[:, 'target']

    return features, target

def fill_NA_values(train_data):
    train_data['gender'].fillna('no_gender', inplace=True)
    train_data['genre_ids'].fillna(0, inplace=True)
    train_data['artist_name'].fillna('no_artist', inplace=True)
    train_data['composer'].fillna('no_composer', inplace=True)
    train_data['lyricist'].fillna('no_lyricist', inplace=True)
    train_data['language'].fillna(0, inplace=True)
    train_data['name'].fillna('no_name', inplace=True)
    train_data['song_length'].fillna(0, inplace=True)
    train_data['isrc'].fillna('no_isrc', inplace=True)
    # 'source_system_tab', 'source_screen_name', 'source_type',
    train_data['source_system_tab'].fillna('no_system_tab', inplace=True)
    train_data['source_screen_name'].fillna('no_screen_name', inplace=True)
    train_data['source_type'].fillna('no_type', inplace=True)
    return train_data


def extract_features(train_data):

    # filter age to remove outliers
    train_data['age'] = train_data['bd'].apply(lambda x: np.int8(x) if (x > 0 and x <= 80) else 0)

    # extract year, month, day from date features
    train_data['registration_init'] = pd.to_datetime(train_data['registration_init_time'],format='%Y%m%d')
    train_data['registration_year'] = pd.DatetimeIndex(train_data['registration_init']).year
    train_data['registration_month'] = pd.DatetimeIndex(train_data['registration_init']).month
    train_data['registration_year'] = pd.DatetimeIndex(train_data['registration_init']).day

    train_data['expiration'] = pd.to_datetime(train_data['expiration_date'],format='%Y%m%d')
    train_data['expiration_year'] = pd.DatetimeIndex(train_data['expiration']).year
    train_data['expiration_month'] = pd.DatetimeIndex(train_data['expiration']).month
    train_data['expiration_year'] = pd.DatetimeIndex(train_data['expiration']).day

    # calculate total duration of membership
    train_data['membership_period'] = train_data['expiration'] - train_data['registration_init']

    # get main genre and count the number of genres
    train_data['main_genre'] = train_data['genre_ids'].apply(lambda x: str(x).split('|')[0])
    train_data['genre_number'] = train_data['genre_ids'].apply(lambda x: len(str(x).split('|')))

    # do the same for artist, composer, lyricist
    train_data['main_artist'] = train_data['artist_name'].apply(lambda x: str(x).split('|')[0])
    train_data['artist_number'] = train_data['artist_name'].apply(lambda x: len(str(x).split('|')))

    train_data['main_composer'] = train_data['composer'].apply(lambda x: str(x).split('|')[0])
    train_data['composer_number'] = train_data['composer'].apply(lambda x: len(str(x).split('|')))

    train_data['main_lyricist'] = train_data['lyricist'].apply(lambda x: str(x).split('|')[0])
    train_data['lyricist_number'] = train_data['lyricist'].apply(lambda x: len(str(x).split('|')))

    # get year of release and country from isrc
    train_data['release_year'] = train_data['isrc'].apply(lambda x: np.int8(str(x)[5:7]) if (x != 'no_isrc') else 0)
    train_data['country'] = train_data['isrc'].apply(lambda x: str(x)[:2] if (x != 'no_isrc') else 'no_country')

    # remove unnecessary columns
    train_data = train_data.drop(columns=['registration_init_time', 'registration_init', 'expiration_date', 'expiration', \
        'genre_ids', 'artist_name', 'composer', 'lyricist', 'isrc', 'bd'], axis=1)
    
    return train_data

def encode_features(train_data, val_data, save_path):
    total_data = pd.concat([train_data, val_data])

    # perform standardization
    for col in train_data.select_dtypes([np.int64,np.float64]):
        # print('standardizing column ', col)
        s = preprocessing.StandardScaler()
        s.fit(total_data[col].to_numpy().reshape(-1,1))
        train_data[col] = s.transform(train_data[col].to_numpy().reshape(-1,1))
        val_data[col] = s.transform(val_data[col].to_numpy().reshape(-1,1))

    # encode all non-numeric features
    for col in train_data.select_dtypes([np.object0, np.timedelta64]):
        # print('encoding column', col)
        l = preprocessing.LabelEncoder()
        l.fit(total_data[col].to_numpy())
        # np.save(save_path+col+'_classes.npy', l.classes_)

        train_data[col] = l.transform(train_data[col].to_numpy())
        val_data[col] = l.transform(val_data[col].to_numpy())
    
    return train_data, val_data

def get_plots(train_data, valid_data):
    # generating tables and plots for the report
    data = pd.concat([train_data,valid_data])
    print('\nTrain dataset columns:\n')
    print(train_data.info())
    print('\nNull columns:\n')
    print(data.isnull().sum() * 100 / len(train_data))

    print('\nTotal number of users:', data['msno'].nunique())
    print('Tontal number of songs:', data['song_id'].nunique())

    # user cold-start
    f1 = plt.figure()
    song_count = data.groupby('msno').count()['song_id'] # song count per user
    print('\nSong count per user: mean {:.2f}, max {}, min {}'.format(song_count.mean(), song_count.max(), song_count.min()))
    song_count.hist(bins=100, range=[0,1000])
    plt.xlabel('song count per user')
    plt.ylabel('number of users')
    plt.savefig('./plots/user_coldstart.png')
    print('Number of users with < 10 song listening history: ', song_count.where(song_count<10).count())
    # song cold-start
    f2 = plt.figure()
    user_count = data.groupby('song_id').count()['msno'] # user count per song
    print('\nUser count per song: mean {:.2f}, max {}, min {}'.format(user_count.mean(), user_count.max(), user_count.min()))
    user_count.hist(bins=40, range=[0,200])
    plt.xlabel('user count per song')
    plt.ylabel('number of songs')
    plt.savefig('./plots/song_coldstart.png')
    print('Number of songs with <= 3 users listened to it: ', user_count.where(user_count<=3).count())   

if __name__ == '__main__':
    main(dataset_path='./datasets/', dev=True)
    # main(dataset_path='./datasets/', dev=False)
