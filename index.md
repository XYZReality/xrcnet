![](images/architecture_figure_improved_compressed.png)

# Code

Our source code is available [here](https://github.com/XYZ-R-D/xrcnet). Please follow the README instructions for downloading the training/testing data, installation and plotting. We provide training/testing pairs for the MegaDepth dataset, as well as our pretrained model. For more information please read the README file.

# Abstract

In this paper, we aim at establishing accurate dense correspondences between a pair of images with overlapping field of view under challenging illumination variation, viewpoint changes, and style differences. Through an extensive ablation study of the state-of-the-art correspondence networks, we surprisingly discovered that the widely adopted 4D correlation tensor and its related learning and processing modules could be de-parameterised and removed from training with merely a minor impact over the final matching accuracy. Disabling some of the most memory consuming and computational expensive modules dramatically speeds up the training procedure and allows to use 4x bigger batch size, which in turn compensates for the accuracy drop. Together with a multi-GPU inference stage, our method facilitates the systematic investigation of the relationship between matching accuracy and up-sampling resolution of the native testing images from 720p to 4K. This leads to finding an optimal resolution 𝕏 that produces accurate matching performance surpassing the state-of-the-art methods particularly over the lower error band for the proposed network and evaluation datasets.

# Bibtex

If you consider using our code/data please consider citing us as follows:

```
@inproceedings{tinchev2020xrcnet, 
    title={ {$\mathbb{X}$}Resolution Correspondence Networks}, 
    author={Tinchev, Georgi and Li, Shuda and Han, Kai and Mitchell, David and Kouskouridas, Rigas}, 
    booktitle={arXiv preprint arXiv:2012.09842},
    year={2020} 
}

```

# Updates

## 02/01/2021 - Released code