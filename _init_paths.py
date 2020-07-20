import os.path as osp
import sys
import os

if os.listdir('data/cache/'):
    os.system('rm data/cache/*')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path)

vg_path = osp.join(this_dir, 'data', 'vgapi')
add_path(vg_path)
