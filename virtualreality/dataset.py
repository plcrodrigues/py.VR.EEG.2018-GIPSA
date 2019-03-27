import mne
import os
import glob
import numpy as np
import datetime as dt
from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
from scipy.io import loadmat
import zipfile

VIRTUALREALITY_URL = 'https://sandbox.zenodo.org/record/261669/files/'


class VirtualReality(BaseDataset):
    '''
    We describe the experimental procedures for a dataset that we have made publicly
    available at https://doi.org/10.5281/zenodo.2605204 in mat (Mathworks, Natick, USA)
    and csv formats. This dataset contains electroencephalographic recordings on 21 
    subjects doing a visual P300 experiment on PC (personal computer) and VR (virtual
    reality). The visual P300 is an event-related potential elicited by a visual 
    stimulation, peaking 240-600 ms after stimulus onset. The experiment was designed 
    in order to compare the use of a P300-based brain-computer interface on a PC and 
    with a virtual reality headset, concerning the physiological, subjective and 
    performance aspects. The brain-computer interface is based on electroencephalography
    (EEG). EEG data were recorded thanks to 16 electrodes. The virtual reality headset 
    consisted of a passive head-mounted display, that is, a head-mounted display which 
    does not include any electronics at the exception of a smartphone. A full description
    of the experiment is available at https://hal.archives-ouvertes.fr/hal-02078533. 
    This experiment was carried out at GIPSA-lab (University of Grenoble Alpes, CNRS,
    Grenoble-INP) in 2018, and promoted by the IHMTEK Company (Interaction Homme-Machine
    Technologie).The study was approved by the Ethical Committee of the University of 
    Grenoble Alpes (Comité d’Ethique pour la Recherche Non-Interventionnelle). 
    The ID of this dataset is VR.EEG.2018-GIPSA.

    **Full description of the experiment and dataset**
    https://hal.archives-ouvertes.fr/hal-02078533

    **Link to the data**
    https://doi.org/10.5281/zenodo.2605204
 
    **Authors**
    Principal Investigator: Eng. Grégoire Cattan
    Technical Supervisors: Eng. Anton Andreev, Eng. Pedro L. C. Rodrigues
    Scientific Supervisor: Dr. Marco Congedo

    **ID of the dataset**
    VR.EEG.2018-GIPSA
    '''

    def __init__(self, VR=True, PC=False):
        super().__init__(
            subjects=list(range(1, 20+1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code='Virtual Reality dataset',
            interval=[0, 1.0],
            paradigm='p300',
            doi='')

        self.VR = VR
        self.PC = PC

    def _get_single_subject_data(self, subject):

        file_path_list = self.data_path(subject)
        sessions = {}
        for filepath in file_path_list:
            session_name = filepath.split('.')[0].split('_')[-1]

            data = loadmat(filepath)['data']

            chnames = ['Fp1', 'Fp2', 'Fc5', 'Fz', 'Fc6', 'T7', 'Cz', 'T8',
                       'P7','P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
            colnames = ['Time'] + chnames + ['Event', 'IsTarget',
            'IsNonTarget', 'IsStartOfNewBlock', 'EndOfRepetitionNumber']

            S = data[:, 1:17]
            stim = 2 * data[:, 18] + 1 * data[:, 19]
            chtypes = ['eeg'] * 16 + ['stim']
            X = np.concatenate([S, stim[:, None]], axis=1).T

            info = mne.create_info(ch_names=chnames + ['stim'], sfreq=512,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)

            idx_stim = np.where(stim > 0)[0]
            idx_blockStart = np.where(data[:,20] > 0)[0]
            idx_repetEndin = np.where(data[:,21] > 0)[0]

            sessions[session_name] = {}
            for bi, idx_bi in enumerate(idx_blockStart):
                start = idx_bi
                end = idx_repetEndin[4::5][bi]
                Xbi = X[:,start:end]

                idx_repetEndin_local = idx_repetEndin[bi*5:(bi*5+5)] - idx_blockStart[bi]
                idx_repetEndin_local = np.concatenate([[0], idx_repetEndin_local])
                for j in range(5):
                    start = idx_repetEndin_local[j]
                    end = idx_repetEndin_local[j+1]
                    Xbij = Xbi[:,start:end]
                    raw = mne.io.RawArray(data=Xbij, info=info, verbose=False)
                    sessions[session_name]['block_' + str(bi+1) + '-repetition_' + str(j+1)] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        file_path_list = []
        if self.VR:
            url = '{:s}subject_{:02d}_VR.mat'.format(VIRTUALREALITY_URL, subject)
            file_path = dl.data_path(url, 'VIRTUALREALITY')
            file_path_list.append(file_path)
        elif self.PC:
            url = '{:s}subject_{:02d}_PC.mat'.format(VIRTUALREALITY_URL, subject)
            file_path = dl.data_path(url, 'VIRTUALREALITY')
            file_path_list.append(file_path)
        return file_path_list
