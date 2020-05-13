# Convolutional neural network to distinguish *Diphasiastrum* taxa and to predict landmarks.

This file explains in which order the JupyterLab files need to be executed and how.

At first, JupyterLab needs to be installed as well as the programming language Python.

For both approaches the file *get_scale_data.ipynb* needs to be executed in advance or the file *all_scale_data.npy* needs to be present.
Additionally, the folder *Utilities* needs to be present in the folders *Classification* and *Landmarks* with the files *affine_math_functions.py* and *preprocessing_functions.py*.

In addition, the folder *DIP_images_fresh* (get the folder from http://bioinf.uni-greifswald.de/bioinf/downloads/data/Diphasiastrum/Diphasiastrum.tar.gz) needs to be present with subfolders 
- *all*                                            
- *images dorsal_2018*
- *images ventral_2018*


as well as the folder *DIP_fresh_data* with the subfolders 
- *dorsal_2018*
- *ventral_2018*

Both approaches have to be handled seperately.
In order to start the classification approach, please only use contents out of the *Classification* folder and for the landmark approach only contents out of the *Landmark* folder are relevant.

However, the execution of the programs of each approach follows identical instructions.
For both approaches, three JupyterLab files exist in the respective folders *Classification* and *Landmarks*.

Classification:
 
1. *classification_image_preprocessing.ipynb*
2. *classification_model_preprocess.ipynb*
3. *classification_model.ipynb*

Landmark prediction:

1. *landmarks_image_preprocessing.ipynb*
2. *landmarks_model_preprocess.ipynb*
3. *landmarks_model.ipynb*

Before being able to execute the files, all packages need to be installed that are mentioned in the Import sections.

It is important to follow the order of execution: 

At first, 1. needs to be executed to preprocess the images. 
Therefore, the hyperparameters need to be set if they want to be changed. 
Possible mistakes in folder names or other aspects concerning the hyperparamteres are taken care of by error messages.
If an error occurs, please correct them.
Otherwise, all cells can be run simultaneously.

Secondly, 2. has to be executed to complete all preparations for the model.
Therefore, the hyperparameters need to be set again only if they want to be changed.
Again, possible mistakes in folder names or other aspects concerning the hyperparamteres are taken care of by error messages.
Apart from these considerations, all cells can be run one after another.

At last, 3. has to be executed to train the model.
Therefore, if different hyperparameters for the model are wanted, the hyperparameters need to be changed. 
Possible mistakes in folder names or other aspects concerning the hyperparamteres are taken care of by error messages.
After that, the cells can be executed one after another.
While training, tensorboard is used to follow the training process via the bash command `tensorboard --logdir=PATH TO LOG FILE`. 


All files and results of my training runs can be found for both approaches in the folders 
- *images* (the folder *images_APPROACH* includes all preprocessed images and the folders *test*, *train*, *val* include the images assigned to the respective datasets), 
- *files* (*.pkl* files that are generated in the respective *APPROACH_model_preprocess.ipynb* files), 
- *model_checkpoints* (final *.h5* file) and 
- *logs* (to see the training curves on tensorboard)

(The *test*, *train*, *val* folder of both approaches can be downloaded at http://bioinf.uni-greifswald.de/bioinf/downloads/data/Diphasiastrum/Diphasiastrum.tar.gz) 
