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
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar']
    makedirs(path)
    for url in _DOWNLOAD_URLS:
        filename = download(url, path=path)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)

if __name__ == '__main__':
    path = os.path.abspath('../../data/pascalvoc/')

    download_voc(path)
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
    shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
    shutil.rmtree(os.path.join(path, 'VOCdevkit'))