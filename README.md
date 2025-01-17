[![DOI:10.3389/fmars.2023.1280510](http://img.shields.io/badge/DOI-<10.3389/fmars.2023.1280510>-<#00000>.svg)](https://doi.org/10.3389/fmars.2023.1280510)

# DeepUVP
# First run with ZooScann Images

# Installation Guide
https://pytorch.org/get-started/locally/

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

```
brew install python@3.12
pip3 install torch torchvision torchaudio
pip3 install -r requirements_.txt
```

# Usage
TBA

# Training - Data needed and computing power

Training: Run main.py

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

```
@software{kronbergapplicationsdeeploki,
  title={DeepLOKI- A deep learning based approach to identify Zooplankton taxa on high-resolution images from the optical plankton recorder LOKI},
  author={Kronberg, Raphael Marvin and Oldenburg, Ellen}
  year = {2023},
  url = {https://github.com/rakro101/DeepLOKI},
}
```
