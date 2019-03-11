import numpy as np
import pandas as pd

def get_block_repetition(X, labels, meta, block_list, repetition_list):

    # TODO : Make the meta_select selection easier and simples with Pandas

    X_select = []
    labels_select = []
    meta_select = []
    for block in block_list:
      for repetition in repetition_list:
          X_select.append(X[meta['run'] == 'block_' + str(block) + '-repetition_' + str(repetition)])
          labels_select.append(labels[meta['run'] == 'block_' + str(block) + '-repetition_' + str(repetition)])
          meta_select.append(meta[meta['run'] == 'block_' + str(block) + '-repetition_' + str(repetition)])
    X_select = np.concatenate(X_select)
    labels_select = np.concatenate(labels_select)
    meta_select = np.concatenate(meta_select)
    df = pd.DataFrame(meta_select, columns=meta.columns)
    meta_select = df

    return X_select, labels_select, meta_select
