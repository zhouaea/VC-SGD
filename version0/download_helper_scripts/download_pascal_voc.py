"""Prepare PASCAL VOC datasets"""
import os
import shutil
import argparse
import tarfile
from gluoncv.utils import download, makedirs
#####################################################################################
# Download and extract VOC datasets into ``path``

def download_voc(path):
    _DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=path, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


if __name__ == '__main__':
    path = os.path.abspath('../../data/pascalvoc/')

    download_voc(path)
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
    shutil.rmtree(os.path.join(path, 'VOCdevkit'))