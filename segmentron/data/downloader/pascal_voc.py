"""Prepare PASCAL VOC datasets"""
from segmentron.utils import download, makedirs
import tarfile
import shutil
import argparse
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]
sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL VOC dataset.',
        epilog='Example: python pascal_voc.py --download-dir ~/VOCdevkit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str,
                        default=None, help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true',
                        help='disable automatic download if set')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite downloaded files if set, in case they are corrupted')
    args = parser.parse_args()
    return args


#####################################################################################
# Download and extract VOC datasets into ``path``

def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    makedirs(path)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(
            url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


#####################################################################################
# Download and extract the VOC augmented segmentation dataset into ``path``

def download_aug(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz',
         '7129e0a480c2d6afb02b517bb18ac54283bfaa35')]
    makedirs(path)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(
            url, path=path, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)
            shutil.move(os.path.join(path, 'benchmark_RELEASE'),
                        os.path.join(path, 'VOCaug'))
            filenames = ['VOCaug/dataset/train.txt', 'VOCaug/dataset/val.txt']
            # generate trainval.txt
            with open(os.path.join(path, 'VOCaug/dataset/trainval.txt'), 'w') as outfile:
                for fname in filenames:
                    fname = os.path.join(path, fname)
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)


if __name__ == '__main__':
    args = parse_args()

    default_dir = os.path.join(root_path, 'datasets/voc')
    if args.download_dir is not None:
        path = args.download_dir
    else:
        path = default_dir
    if not os.path.isfile(path) or not os.path.isdir(os.path.join(path, 'VOC2007')) \
            or not os.path.isdir(os.path.join(path, 'VOC2012')):
        if args.no_download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you should not disable "--no-download" to grab it'.format(path)))
        else:
            download_voc(path, overwrite=args.overwrite)
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'),
                        os.path.join(path, 'VOC2007'))
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'),
                        os.path.join(path, 'VOC2012'))
            shutil.rmtree(os.path.join(path, 'VOCdevkit'))

    if not os.path.isdir(os.path.join(path, 'VOCaug')):
        if args.no_download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you should not disable "--no-download" to grab it'.format(path)))
        else:
            download_aug(path, overwrite=args.overwrite)

    try:
        os.symlink(path, default_dir)
    except Exception as e:
        print(e)
