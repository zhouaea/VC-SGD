# For installation on a VM with Ubuntu 20.04 and Python 3.5
# Copy paste this:
#   git clone https://github.com/zhouaea/VC-SGD.git
#   cd VC-SGD/
#   git checkout object_detection
#   cd version0/download_helper_scripts/
#   . install.sh


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
python3.9 -m pip install --ignore-installed -r requirements.txt

# Optimize cpu performance
export KMP_AFFINITY=granularity=fine,compact,1,0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 2))

# Install pascalvoc dataset in VC-SGD/data/pascalvoc
python3.9 download_pascal_voc.py
rm ../../data/pascalvoc/*.tar
cd ..

# Download application that will continue to run experiments even if the ssh connection breaks.
sudo apt-get install screen