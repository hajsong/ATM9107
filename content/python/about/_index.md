+++
title = "Python"
date = 2019-03-05T09:48:48+09:00
weight = 1
chapter = false
pre = "<b>1. </b>"
+++


### A quick introduction to python and its packages

For more detailed information, see [here](https://currents.soest.hawaii.edu/ocn760_4/basics.html)  
For a quick start, refer [here](https://learnxinyminutes.com/docs/python/)

### Introduction

+ Python has been gaining solid ground in earth sciences community.
+ Python is light and efficient, while is able to do most of things that Matlab offers.

During the last couple of decades, Matlab has been the most commonly-used scripting language in physical oceanography, and it has a large user base in many other fields. Recently, however, Python has been gaining ground, often being adopted by former Matlab users as well as by newcomers. Here is a little background to help you understand this shift, and why we advocate using Python from the start.

### Matlab and Python

|    Matlab                              |        Python                                |
|:--------------------------------------:|:--------------------------------------------:|
|  expensive                             |               Free                           |
|  Well-documented                       |       Major packages are well-documented     |
| A wide range of functions is included  |  Packages needs to be installed individually |
| Possibly more consistency within it    |  Possible inconsistency between packages     |
|    fast                                |    Faster!!!                                 |

### Why use Python instead of Matlab?

Python is fundamentally a better computer language in many ways.
It is suitable for a wider variety of tasks.
It scales better from the shortest of scripts to large software projects.
It facilitates writing clearer and more concise code.
With associated tools, it makes for easier access to existing high-performance codes in compiled languages, and for using smaller pieces of compiled code to speed up critical sections.
Because Python is Free and Open Source Software (FOSS), you can install it on any machine without having to deal with a license manager.
For the same reason, Python code that is part of a research project can be run by anyone, anywhere, to verify or extend the results.
Most Python packages you are likely to want to use are developed in an open environment. The scientific Python ecosystem is dynamic and friendly.

### What are the potential disadvantages?

Installation of all the packages one needs can take time and expertise; but distributions like Anaconda, combined with other improvements in python packaging software and repositories, are rapidly solving this problem.
Although progress is being made, the scientific Python stack is not as uniformly well-documented as Matlab; it might take longer to figure out how to do something in Python. You might also find that a routine available in Matlab is not yet available in a Python package.
Matlab is still mainstream in oceanography–at least among many of the old guard; with Python, you are an early adopter. (If you have a spirit of adventure, this might be considered an advantage.)

Matlab evolved from a simple scripting language designed to provide interactive access to a public-domain linear algebra library. It has since grown into a more sophisticated but still specialized language with an extensive set of routines written, organized, and documented by one of the great successes in the software world, Mathworks.

Python, in contrast, was designed by a computer scientist as a general-purpose scripting language for easy adoption and widespread use. People tried it and liked it, and the result is that it is widely used throughout the software world, for all sorts of tasks, large and small. There is a vast array of Python packages that arem freely available to do all sorts of things—including the sorts of things that oceanographers and other scientists do; but these packages are not neatly bound up in a single product, and the documentation for the language itself and for the packages is similarly scattered and of varying quality.

### The Python scientific stack for oceanography

#### Essentials

+ numpy -
N-dimensional arrays and functions for working with them. This is the heart of the scientific stack for Python.

+ matplotlib -
A two-dimensional plotting library with a Matlab-like interface.

+ IPython -
An interactive shell, and much more.

+ jupyter notebook -
A web interface to IPython, with integrated plots and rich text, including equations. (It can also be used with languages like R and Julia.)

+ scipy -
A large number of functions for statistics, linear algebra, integration, etc. It also includes functions for reading and writing Matlab’s binary matfiles.

#### Other packages of interest

+ pandas -
Data structures inspired by R (Series, DataFrame, Panel), with associated functionality for fast and sophisticated indexing, subsetting, and calculations.

+ Cython -
A language with augmented Python syntax for writing compiled Python extensions. I use it for writing interfaces to code written in C, and for writing extensions to speed up critical operations.

+ basemap -
Matplotlib toolkit for making maps with any of a wide variety of projections. Matplotlib’s full plotting capabilities can then be used on the map.

+ netCDF4 -
Python interface to the NetCDF libraries, supporting the new version 4 in addition to the widespread version 3, and also supporting access to datasets via OPeNDAP.
