



wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
rm -rf wiki-news-300d-1M-subword.vec.zip

python data_preprocess.py
python calc_cosine_distance.py
