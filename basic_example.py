
import numpy as np
import mne
from virtualreality.dataset import VirtualReality
from virtualreality.utilities import get_block_repetition

import moabb
from moabb.paradigms import P300

from pyriemann.estimation import ERPCovariances
from pyriemann.classification import MDM

import warnings
warnings.filterwarnings("ignore")

# define the dataset instance
dataset = VirtualReality(VR=False, PC=True)

# get the paradigm
paradigm = P300()

# get the epochs and labels
X, labels, meta = paradigm.get_data(dataset, subjects=[1])

# # example of how to select the blocks and repetitions of interest
# block_list = [1, 2, 3]
# repetition_list = [1]
# X_select, labels_select, meta_select = get_block_repetition(X, labels, meta, block_list, repetition_list)
#
# # split in training and testing datasets
# block_training = list(range(1, 10+1))
# repetition_train = [1, 2, 3]
# X_training, labels_training, meta_training = get_block_repetition(X, labels, meta, block_training, repetition_train)
#
# block_test = [11, 12]
# repetition_test = [1]
# X_test, labels_test, meta_test = get_block_repetition(X, labels, meta, block_test, repetition_test)
# repetition_test = [1, 2]
# X_test2, labels_test2, meta_test2 = get_block_repetition(X, labels, meta, block_test, repetition_test)
# repetition_test = [1, 2, 3]
# X_test3, labels_test3, meta_test3 = get_block_repetition(X, labels, meta, block_test, repetition_test)
#
# # estimate the extended ERP covariance matrices
# erpc = ERPCovariances(classes=['Target'])
# erpc.fit(X_training, labels_training)
# covs_training = erpc.transform(X_training)
# covs_test = erpc.transform(X_test)
# covs_test2 = erpc.transform(X_test2)
# covs_test3 = erpc.transform(X_test3)
#
# # get the classification accuracy (TODO : change to AUC for P300)
# clf = MDM()
# clf.fit(covs_training, labels_training)
# scr = clf.score(covs_test, labels_test)
# scr2 = clf.score(covs_test2, labels_test2)
# scr3 = clf.score(covs_test3, labels_test3)
# print(scr, scr2, scr3)

# plotting the P300 signals
import matplotlib.pyplot as plt
chi = 6
fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
ax.plot(np.linspace(0, 1, 513), np.mean(X[labels == 'Target'], axis=0)[chi,:], color='b')
ax.plot(np.linspace(0, 1, 513), np.mean(X[labels == 'NonTarget'], axis=0)[chi,:], color='r')
fig.show()
