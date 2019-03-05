+++
title = "Installation"
date = 2019-03-05T09:52:17+09:00
weight = 2
chapter = false
pre = "<b>2. </b>"
+++

The instructions are based on using the Anaconda software distribution, which provides platform-independent package management for Python and other software in self-contained user-specific environments.


### 1. Acquire Anaconda (or Miniconda)
**This is highly recommended.**

Download the Anaconda software from [https://www.anaconda.com/download/#macos](https://www.anaconda.com/download/#macos)  
Document for the software can be found [here](https://conda.io/docs/user-guide/install/download.html?highlight=miniconda#anaconda-or-miniconda)

If you choose to go with something light, you may consider getting [miniconda] (https://conda.io/miniconda.html).


#### Anaconda or Miniconda? (from conda manual)

Choose Anaconda if you:

+ Are new to conda or Python.
+ Like the convenience of having Python and over 150 scientific packages automatically installed at once.
+ Have the time and disk space—a few minutes and 300 MB.
+ Do not want to individually install each of the packages you want to use.

Choose Miniconda if you:

+ Do not mind installing each of the packages you want to use individually.
+ Do not have time or disk space to install over 150 packages at once.
+ Want fast access to Python and the conda commands and you wish to sort out the other programs later.

### 2. Verify that you have python


```
$ which python
$ python --version
```
which dump out the following output in my case:
```
/Users/hajsong/anaconda3/bin/python
Python 3.7.0
```

If you want to install different version of python (e.g. python 3.6), you can create a new environment where python 3.6 becomes default version.

```
conda create -n py36 python=3.6 anaconda
```

To activate this environment, use:

```
source activate py36
```

To deactivate an active environment, use:
```
source deactivate
```
In window,
```
activate myenv
deactivate
```

If you want to remove the environment 'py36', try
```
conda remove --name py36 --all
```

### 3. Install packages

It is easy to manage (install and update) packages with Anaconda (or Miniconda).
For instance, if you want to install numpy package,

```
conda install numpy
```

If you want to update numpy package,

```
conda update numpy
```

### 4. Ipython and Jupyter notebook

[Ipython](https://ipython.org/index.html) provides an interactive environment for doing python.  
It is included in the Anaconda software and you can visit its [document page](http://ipython.readthedocs.io/en/stable/).  

If you want to launch Jupyter notebook in one of your environments (e.g. py36), activate that environment first before starting Jupyter notebook.

To launch ipython, just do  

```
ipython
```  

To launch Jupyter notebook, type  
```
jupyter notebook
```

### 5. Jupyter notebook on the remote server
You might experience the delay of bringing the figure that you made remotely up on your local machine.
This pain can be alleviated if you run Jupyter notebook on the remote server.
Instead of opening up the web browser on the remote server, you can request the export the content to the port.

To do this, launch Jupyter notebook on the remote server as the following.
```
jupyter notebook --no-browser --port=8889
```
You can change the number for the port.

Then, on your local machine, connect to your server and receive the data by typing the following command.
```
ssh -N -f -L localhost:8899:localhost:8889 [yourid]@[server_address]
```
Make sure that the second localhost number match the one on the remote server.
Then, open up the web browser and go to "localhost:8899" by typing it in the address section.
The web browser might ask the password, and it appears on the terminal where you launch Jupyter notebook.
