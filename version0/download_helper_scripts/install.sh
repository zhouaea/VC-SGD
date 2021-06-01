# For installation on a VM with Ubuntu 16.04 and Python 3.5

# Download Python 3.9
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9
python3.9 --version

# Download pip for python 3.9
sudo apt install python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py

# Install dependencies
pip install -r requirements.txt

# Install pascalvoc dataset in VC-SGD/data/pascalvoc
python3.9 download_pascal_voc.py
rm ../../data/pascalvoc/*.tar