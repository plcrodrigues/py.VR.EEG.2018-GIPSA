
from virtualreality.dataset import VirtualReality
from virtualreality.utilities import get_block_repetition

import numpy as np
import mne
import moabb
import pandas as pd

from moabb.paradigms import P300
from pyriemann.estimation import ERPCovariances
from pyriemann.classification import MDM

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib

from tqdm import tqdm

"""
=============================
Classification of the trials
=============================

This example shows how to extract the epochs from the dataset of a given
subject and then classify them using Riemannian Geometry framework for BCI. 
We compare the scores in the VR and PC conditions.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# create dataset
dataset = VirtualReality()

# get the paradigm
paradigm = P300()

# loop to get scores for each subject
nsubjects = 5

df = {}
for tmax in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

	paradigm.tmax = tmax

	scores = []
	for subject in tqdm(dataset.subject_list[:nsubjects]):
		scores_subject = [subject]
		for condition in ['VR', 'PC']:

			# define the dataset instance
			if condition == 'VR':
				dataset.VR = True
				dataset.PC = False
			elif condition == 'PC':
				dataset.VR = False
				dataset.PC = True

			# get the epochs and labels
			X, labels, meta = paradigm.get_data(dataset, subjects=[subject])
			labels = LabelEncoder().fit_transform(labels)

			kf = KFold(n_splits = 6)
			repetitions = [1, 2]				
			auc = []

			blocks = np.arange(1, 12+1)
			for train_idx, test_idx in kf.split(np.arange(12)):

				# split in training and testing blocks
				X_training, labels_training, _ = get_block_repetition(X, labels, meta, blocks[train_idx], repetitions)
				X_test, labels_test, _ = get_block_repetition(X, labels, meta, blocks[test_idx], repetitions)

				# estimate the extended ERP covariance matrices with Xdawn
				dict_labels = {'Target':1, 'NonTarget':0}
				erpc = ERPCovariances(classes=[dict_labels['Target']], estimator='lwf')
				erpc.fit(X_training, labels_training)
				covs_training = erpc.transform(X_training)
				covs_test = erpc.transform(X_test)

				# get the AUC for the classification
				clf = MDM()
				clf.fit(covs_training, labels_training)
				labels_pred = clf.predict(covs_test)
				auc.append(roc_auc_score(labels_test, labels_pred))

			# stock scores
			scores_subject.append(np.mean(auc))

		scores.append(scores_subject)

	# print results
	df[tmax] = pd.DataFrame(scores, columns=['subject', 'VR', 'PC'])

filename = './results.pkl'
joblib.dump(df, filename)
