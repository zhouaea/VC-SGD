# For installation on a BC VM with Ubuntu 28.04 and Python 3.6
# Copy paste this:
#   git clone https://github.com/zhouaea/VC-SGD.git
#   cd VC-SGD/
#   git checkout object_detection
#   cd version0/download_helper_scripts/
#   . install.sh

# Make sure this is run in the download_helper_scripts directory.

# Install dependencies
python3 -m pip install -r requirements.txt

# Install pascalvoc dataset in VC-SGD/data/pascalvoc
python3 download_pascal_voc.py
rm ../../data/pascalvoc/*.tar
cd ..

# Download application that will continue to run experiments even if the ssh connection breaks.
#apt-get install screen
