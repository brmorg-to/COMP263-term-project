# Deep Learning for Real-World Problems

## Project Overview
This project applies advanced deep learning techniques to solve a real-world problem using both supervised and unsupervised learning models. Our focus is on image processing tasks related to brain tumor detection from MRI scans using Convolutional Neural Networks (CNN) and Generative Adversarial Networks (GANs).

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Running the Code](#running-the-code)
- [Models and Training](#models-and-training)
- [Results](#results)
- [Contributors](#contributors)

## Setup Instructions
To run this project, you need to have Python and Conda installed on your machine. Follow these steps to set up the environment:

1. **Clone the Repository**
cd your-project-directory
git clone [PROJECT](https://github.com/brmorg-to/COMP263-term-project.git)

2. **Create and Activate the Conda Environment**
Use the `environment.yml` file to create a Conda environment that replicates the exact setup used for development. This environment will include all necessary packages.
```bash
conda env create -n conda-env -f /path/to/environment.yml
conda activate /path/to/conda-env'
```

## Running the Code
Once the environment is set up, you can run the code by navigating to the source code directory and executing the main script:
```bash
cd src
python concatenated_classifier.py
```

## Models and Training
- Supervised Learning Model: A Vision Transformer (ViT) model fine-tuned for brain tumor classification. This model leverages pre-trained weights and has been adapted specifically to handle the nuances of MRI brain scans.
- Unsupervised Learning Model: A Generative Adversarial Network (GAN) used to generate additional synthetic images to augment the training data for the supervised model.

## Contributors

- Christian Aduna
- Lance Nelson
- My Duyen Phung
- Bruno Morgado

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE.md) - see the LICENSE.md file for details.
