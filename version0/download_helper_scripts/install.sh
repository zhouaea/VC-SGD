# For installation on a VM with Ubuntu 20.04 and Python 3.5
# Copy paste this:
#   git clone https://github.com/zhouaea/VC-SGD.git
#   cd VC-SGD/
#   git checkout object_detection
#   cd version0/download_helper_scripts/
#   . install.sh

# Make sure this is run in the download_helper_scripts directory.

# Download Python 3.9
apt update
apt install software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.9
python3.9 --version

# Download pip for python 3.9
apt install python3.9-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py

# Install dependencies
python3.9 -m pip install requirements.txt

# Install pascalvoc dataset in VC-SGD/data/pascalvoc
python3.9 download_pascal_voc.py
rm ../../data/pascalvoc/*.tar
cd ..

# Download application that will continue to run experiments even if the ssh connection breaks.
#apt-get install screen
