# EEG Virtual Reality dataset

Repository with basic scripts for using the EEG Virtual Reality developed at GIPSA-lab [1]. The dataset files and their documentation are all available at 

[https://sandbox.zenodo.org/record/261669/files/](https://sandbox.zenodo.org/record/261669/files/)

The code of this repository was developed in **Python 3** using MOABB [2] and MNE-Python [3, 4] as tools for downloading and pre-processing the dataset. 

To make things work, you will need to install MOABB. It can be done by following the instructions at the project's GitHub [link](https://github.com/NeuroTechX/moabb).

You might also need to install some packages. They are all listed in the `requirements.txt` file and can be easily installed by doing

```
pip install -r requirements.txt
```

in your command line. 

Then, to ensure that your code finds the right scripts whenever you do `import virtualreality`, you should also do

```
python setup.py develop # because no stable release yet
```

Note that you might want to create a *virtual environment* before doing all these installations.

# References

[1] Cattan et al. "Dataset of an EEG-based BCI experiment in Virtual Reality and on a Personal Computer" [DOI](https://hal.archives-ouvertes.fr/hal-02078533)

[2] Jayaram and Barachant "MOABB: Trustworthy algorithm benchmarking for BCIs" [DOI](https://doi.org/10.1088/1741-2552/aadea0)

[3] Gramfort et al. "MNE software for processing MEG and EEG data" [DOI](https://doi.org/10.1016/j.neuroimage.2013.10.027)

[4] Gramfort et al. "MEG and EEG data analysis with MNE-Python" [DOI](https://doi.org/10.3389/fnins.2013.00267)

