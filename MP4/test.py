import os
import os.path
import hashlib


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        print('1')
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        print('2')
        return False
    return True


PATH_TO_CIFAR100_CS543 = "./"
FILENAME = "cifar-100-cs543-python.tar"
tgz_md5 = 'e68a4c763591787a0b39fe2209371f32'
fpath = os.path.join(PATH_TO_CIFAR100_CS543, FILENAME)
print('fpath =', fpath)
print('tgz_md5 =', tgz_md5)
print(check_integrity(fpath, tgz_md5))


