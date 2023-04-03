# Assuming script run from top level directory
# Download data from https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz
cd ./data
wget https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz -O ./frames.tar.gz
tar -xvzf ./frames.tar.gz -C ./raw
rm ./frames.tar.gz

# Change directory to ./data/raw
cd ./raw

# Unzip the file metadata.csv.gz
gunzip metadata.csv.gz

# Change current directory to ../../
cd ../../