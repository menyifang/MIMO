chumpy
======

[![version](https://img.shields.io/pypi/v/chumpy?style=flat-square)][pypi]
[![license](https://img.shields.io/pypi/l/chumpy?style=flat-square)][pypi]
[![python versions](https://img.shields.io/pypi/pyversions/chumpy?style=flat-square)][pypi]
[![build status](https://img.shields.io/circleci/project/github/mattloper/chumpy/master?style=flat-square)][circle]

Autodifferentiation tool for Python.

[circle]: https://circleci.com/gh/mattloper/chumpy
[pypi]: https://pypi.org/project/chumpy/


Installation
------------

Install the fork:

```sh
pip install chumpy
```

Import it:

```py
import chumpy as ch
```

Overview
--------

Chumpy is a Python-based framework designed to handle the **auto-differentiation** problem,
which is to evaluate an expression and its derivatives with respect to its inputs, by use of the chain rule.

Chumpy is intended to make construction and local
minimization of objectives easier.

Specifically, it provides:

- Easy problem construction by using Numpy’s application interface
- Easy access to derivatives via auto differentiation
- Easy local optimization methods (12 of them: most of which use the derivatives)


Usage
-----

Chumpy comes with its own demos, which can be seen by typing the following:

```python
import chumpy
chumpy.demo() # prints out a list of possible demos
```


License
-------

This project is licensed under the MIT License.
