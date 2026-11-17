mkdir -p benchs/multilayer/data

curl --fail --location --retry 3 --continue-at - ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz --output benchs/multilayer/data/sift.tar.gz

tar -xzf benchs/multilayer/data/sift.tar.gz -C benchs/multilayer/data

curl --fail --location --retry 3 --continue-at - ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz --output benchs/multilayer/data/gist.tar.gz

tar -xzf benchs/multilayer/data/gist.tar.gz -C benchs/multilayer/data

mkdir -p benchs/multilayer/data/deep

curl --fail --location --retry 3 --continue-at - https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin --output benchs/multilayer/data/deep/base.10M.fbin

curl --fail --location --retry 3 --continue-at - https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin --output benchs/multilayer/data/deep/query.public.10K.fbin

mkdir -p benchs/multilayer/data/glove

curl --fail --location --retry 3 --continue-at - http://ann-benchmarks.com/glove-25-angular.hdf5 --output benchs/multilayer/data/glove/glove-25-angular.hdf5

python benchs/multilayer/convert_glove.py benchs/multilayer/data/glove/glove-25-angular.hdf5 benchs/multilayer/data/glove
