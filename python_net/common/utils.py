

import os


def path_generate(root, subpath):
    path = os.path.join(root, subpath)
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path):
        return path
    return path
