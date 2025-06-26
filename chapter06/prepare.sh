FILE_ID="0B7XkCwpI5KDYNlNUTTlSS21pQmM";
FILE_NAME="test.txt";
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$FILE_ID" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p');
mkdir data
wget -P data --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID" -O $FILE_NAME;
rm -f /tmp/cookies.txt

!wget -P data http://download.tensorflow.org/data/questions-words.txt

!wget -P data https://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
unzip data/wordsim353.zip -d data/
rm data/wordsim353.zip