# Interactive Grabcut

It is my implementation of Grabcut for interactive image segmentation.

<p float="left">
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/Data/test3.jpg" width="300" />
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/Data/test2.jpg" width="300" /> 
  <img src="https://github.com/hsihsun/Interactive-Grabcut/blob/master/grabcut.jpg" width="300" /> 
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

# How to execute:

Example: python3 grabcut.py ./Data/test1.jpg 10 5 2 5 1

# Paremeters:

```
1.Path of image                    (here: Data/test1.jpg)

2.Number of iteration              (here: 10)

3.Number of clusters in GMM        (here: 5)

4.Scaling to resize image          (here: 2)--> image from (H by W) to (H/2 by W/2)

5.Gamma value                      (here: 5)

6.show alpha map or not (1 or 0)   (here: 1)
```
