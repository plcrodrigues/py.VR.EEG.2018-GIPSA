
from virtualreality.dataset import VirtualReality
from virtualreality.utilities import get_block_repetition

import numpy as np
import mne
import moabb
import pandas as pd
import matplotlib.pyplot as plt

from moabb.paradigms import P300
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from pyriemann.classification import MDM
from pyriemann.embedding import Embedding
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from tqdm import tqdm

"""
===============================
Spectral Embedding of the data
===============================

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# get the paradigm
paradigm = P300()
paradigm.tmax = 1.0

# create datasets
datasets = {}
datasets['VR'] = VirtualReality()
datasets['VR'].VR = True
datasets['VR'].PC = False
datasets['PC'] = VirtualReality()
datasets['PC'].VR = False
datasets['PC'].PC = True

results = {}

for subject in VirtualReality().subject_list:

	data = {}
	data['VR'] = {}
	data['PC'] = {}

	embeddings = {}
	stats = {}
	stats['VR'] = {}
	stats['PC'] = {}

	for condition in datasets.keys():

		# get the epochs and labels
		X, y, meta = paradigm.get_data(datasets[condition], subjects=[subject])
		y = LabelEncoder().fit_transform(y)

		data[condition]['X'] = X
		data[condition]['y'] = y

		# estimate xDawn covs
		ncomps = 4
		erp = XdawnCovariances(classes=[1], estimator='lwf', nfilter=ncomps, xdawn_estimator='lwf')
		#erp = ERPCovariances(classes=[1], estimator='lwf', svd=ncomps)	
		split = train_test_split(X, y, train_size=0.50, random_state=42)
		Xtrain, Xtest, ytrain, ytest = split
		covs = erp.fit(Xtrain, ytrain).transform(Xtest)

		Mtarget = mean_riemann(covs[ytest == 1])
		Mnontarget = mean_riemann(covs[ytest == 0])
		stats[condition]['distance'] = distance_riemann(Mtarget, Mnontarget)
		stats[condition]['dispersion_target'] = np.sum([distance_riemann(covi, Mtarget)**2 for covi in covs[ytest == 1]]) / len(covs[ytest == 1])
		stats[condition]['dispersion_nontarget'] = np.sum([distance_riemann(covi, Mnontarget)**2 for covi in covs[ytest == 0]]) / len(covs[ytest == 0])
		
	print('subject', subject)
	print(stats)

	results[subject] = stats

	# covs = np.concatenate([covs, Mtarget[None,:,:], Mnontarget[None,:,:]])
	# ytest = np.concatenate([ytest, [1], [0]])
	# data[condition]['ytest'] = ytest

	# # do the embedding
	# lapl = Embedding(metric='riemann', n_components=2)
	# embeddings[condition] = lapl.fit_transform(covs)

dispersion_target_list = []
dispersion_nontarget_list = []
distance_list = []
condition_list = []
subject_list = []
for subject in results.keys():
	results_subj = results[subject]
	for condition in ['VR', 'PC']:
		subject_list.append(subject)
		condition_list.append(condition)
		dispersion_target_list.append(results_subj[condition]['dispersion_target'])
		dispersion_nontarget_list.append(results_subj[condition]['dispersion_nontarget'])
		distance_list.append(results_subj[condition]['distance'])

df = pd.DataFrame()
df['subject'] = subject_list
df['condition'] = condition_list
df['dispersion_target']	= dispersion_target_list
df['dispersion_nontarget'] = dispersion_nontarget_list
df['distance'] = distance_list

#####

# # plot
# names = {0:'NonTarget', 1:'Target'}
# colors = {0:'#2166ac', 1:'#b2182b'}
# fig, ax = plt.subplots(figsize=(16, 7.4), facecolor='white', ncols=2)
# for axi, condition in zip(ax, embeddings.keys()):

# 	embd = embeddings[condition]

# 	for label in np.unique(data[condition]['ytest']):
# 	    idx = (data[condition]['ytest'] == label)
# 	    axi.scatter(embd[idx, 0], embd[idx, 1], s=80, label=names[label], alpha=0.25, edgecolor='none', facecolor=colors[label])

# 	axi.scatter(embd[-2,0], embd[-2,1], s=150, color=colors[1])
# 	axi.scatter(embd[-1,0], embd[-1,1], s=150, color=colors[0])

# 	axi.plot([ embd[-2,0], embd[-1,0] ], [ embd[-2,1], embd[-1,1] ], c='k', lw=0.8, ls='--')

# 	axi.grid(False)
# 	axi.set_xticks([-1, -0.5, 0, +0.5, 1.0])
# 	axi.set_yticks([-1, -0.5, 0, +0.5, 1.0])
# 	axi.set_title(condition, fontsize=16)
# 	axi.legend()

# fig.savefig('embedding.png', format='png')	
#fig.show()

