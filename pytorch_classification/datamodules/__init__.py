# Split definition of DTD, EuroSAT and SUN397. https://github.com/mlfoundations/task_vectors/issues/1
from .dtd import DTDDataModule
from .eurosat import EuroSATDataModule
from .gtsrb import GTSRBDataModule
from .imagenet import ImageNetDataModule
from .mnist import MNISTDataModule
from .resisc45 import RESISC45DataModule
from .sun397 import SUN397DataModule
from .svhn import SVHNDataModule
from .stanford_cars import StanfordCarsDataModule