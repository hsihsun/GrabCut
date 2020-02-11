# Interactive Grabcut

It is my python implementation of Grabcut for interactive image segmentation.

<p float="left">
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/Data/dog.jpg" width="280" />
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/readme_data/select.jpg" width="280" /> 
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/readme_data/grabcut.jpg" width="280" /> 
</p>


This project implements

    @article{rother2004grabcut,
      title={Grabcut: Interactive foreground extraction using iterated graph cuts},
      author={Rother, Carsten and Kolmogorov, Vladimir and Blake, Andrew},
      journal={ACM Transactions on Graphics (TOG)},
      volume={23},
      number={3},
      pages={309--314},
      year={2004},
      publisher={ACM}
    }

# How to setup and execute:

Please download this repository and install https://github.com/pmneila/PyMaxflow

Example: python3 grabcut.py ./Data/dog.jpg 10 5 2 5 1

# Paremeters:

```
1.Path of image                    (here: Data/test1.jpg)

2.Number of iteration              (here: 10)

3.Number of clusters in GMM        (here: 5)

4.Scaling to resize image          (here: 2)--> image from (H by W) to (H/2 by W/2)

5.Gamma value                      (here: 5)

6.show alpha map or not (1 or 0)   (here: 1)
```
