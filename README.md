[![DOI:10.3389/fmars.2023.1280510](http://img.shields.io/badge/DOI-<10.3389/fmars.2023.1280510>-<#00000>.svg)](https://doi.org/10.3389/fmars.2023.1280510)

# DeepUVP
# First run with ZooScann Images

Repo moved to GitLab:  https://gitlab.com/qtb-hhu/marine/deepuvp



# Installation Guide
https://pytorch.org/get-started/locally/

```sh
brew install python@3.12
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

# Usage
Additional Networks/ Architectures can be implemented in model_arc.
The dataloader is in dataloader.py.
The training logic loop is in loop.py defined.
The hyperparameter dicts in the config.py.

Set up an Weight and Biases Account: https://wandb.ai/site/

# Training - Data needed and computing power
Here i use an open source Zooscan dataset https://www.seanoe.org/data/00446/55741/
But we could use the dataloader from DeepLOKI aswell.

Training: Run main.py
Sweeps/Hyperparameter Tuning:  Run sweeps_main.py

# Software used
Training and Validation was performed on an Nvidia A$100$ (Nvidia Corp., Santa Clara, CA, USA) and on Apple M1 MAX with 32 GB (Apple, USA), depending on the computational power needed, for example self-supervised pre-training was performed on a Hyper performing cluster with Nvidia A$100$. <br>
On the Macbook Pro (Apple, USA) we used:<br>

# Authors
Raphael Kronberg and Ellen Oldenburg

# Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated !

# Contributing to DeepUVP
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. 


# License , citation and acknowledgements
Based on an earlier paper:
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. Please cite our paper, when using this code:

# Dataset dataset_012
Balanced datset, with copies / resample images of classes where we had only a few images.
Annelida, Appendicularia, Artefact, Chaetognatha, Copepoda, Crsutacea, Preropoda, Radiolaria
Train (800 per class) , Val(100 per class), Test Split(100 per class)
(0.8,0.1,0.1)

```
@software{kronbergapplicationsdeeploki,
  title={DeepLOKI- A deep learning based approach to identify Zooplankton taxa on high-resolution images from the optical plankton recorder LOKI},
  author={Kronberg, Raphael Marvin and Oldenburg, Ellen}
  year = {2023},
  url = {https://github.com/rakro101/DeepLOKI},
}
```
