
import sys
import os
import logging
import papermill as pm
import scrapbook as sb
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams
)
from recommenders.datasets.amazon_reviews import download_and_extract, data_preprocessing
from recommenders.datasets.download_utils import maybe_download


from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
####  to use the other model, use one of the following lines:
# from recommenders.models.deeprec.models.sequential.asvd import A2SVDModel as SeqModel
# from recommenders.models.deeprec.models.sequential.caser import CaserModel as SeqModel
# from recommenders.models.deeprec.models.sequential.gru4rec import GRU4RecModel as SeqModel
# from recommenders.models.deeprec.models.sequential.sum import SUMModel as SeqModel

#from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
#from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator
if __name__ == '__main__':
    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))


