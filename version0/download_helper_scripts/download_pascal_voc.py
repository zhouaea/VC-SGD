# NOTE: MAKE SURE TO RUN THIS WITHIN DOWNLOAD_HELPER_SCRIPTS FOLDER
# If there is a connection error, simply cd into download_helper_scripts and run this script again.
# If the script still refuses to run, the host server is probably down. You can verify by googling the website.
import os
import shutil
import argparse
import tarfile
from gluoncv.utils import download, makedirs
#####################################################################################
# Download and extract VOC datasets into ``path``

def download_voc(path):
    _DOWNLOAD_URLS = [
        ('http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


if __name__ == '__main__':
    path = os.path.abspath('../../data/pascalvoc/')
    if not os.path.exists(path):
        os.makedirs(path)

    download_voc(path)
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
    shutil.rmtree(os.path.join(path, 'VOCdevkit'))