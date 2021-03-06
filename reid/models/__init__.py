from __future__ import absolute_import

from .ResNet import *
from .SEResNet import *
from .DenseNet import *
from .hacnn import *
from .mudeep import *
from .hacnn import *
from .pcb import *
from .mlfn import *
from .resnetmid import *
from .mgn import *

__factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnext50': SEResNeXt50,
    'seresnext101': SEResNeXt101,
    'densenet121': DenseNet121,
    'hacnn':HACNN,
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'mgn': MGN
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
