GOOGLE_DRIVE_ID="0B7XkCwpI5KDYNlNUTTlSS21pQmM"
FILE_NAME="GoogleNews-vectors.bin.gz"

source ../.venv/bin/activate

mkdir data
cd data
gdown "$GOOGLE_DRIVE_ID" -O "$FILE_NAME"
gunzip -f "$FILE_NAME"
cd ..
deactivate

wget -P data "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
unzip data/SST-2.zip