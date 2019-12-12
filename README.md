# Interactive Image Segmentation

It is my implementation of Interactive Image Segmentation using GrabCut.


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

Comments: python3 grabcut.py ./Data/test1.jpg 20 5 2 2 1

# Paremeters:
```
1.Path of image                    (here: Data/test1.jpg)

2.Number of iteration              (here: 20)

3.Number of clusters in GMM        (here: 5)

4.Scaling to resize image          (here: 2)--> image from (H by W) to (H/2 by W/2)

5.show alpha map or not            (here: 1)
```
