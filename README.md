# KKBox Music Recommendation

1. Clone the repo
2. Download datasets from Kaggle (you will need to join the competition first https://www.kaggle.com/c/kkbox-music-recommendation-challenge/overview)
```sh
mkdir datasets
cd datasets
kaggle competitions download -c kkbox-music-recommendation-challenge
```
3. Extract compressed files
```sh
7z x kkbox-music-recommendation-challenge.zip
find . -type f -name "*.7z" -execdir 7za x {} \; -exec rm -- {} \;
```
4. Preprocessing
```sh
python3 process_datasets.py
```
5. Training models
```sh
python3 music_pred.py
```

