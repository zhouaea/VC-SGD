py_ver=python3.8                   # the version of Python to be downloaded in install_python

sudo apt update
sudo apt install -y gcc
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y ${py_ver}
python3.8 --version

pip install -r requirements.txt