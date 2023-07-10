matSINDy
========

**matrix Sparse Identification of Non-linear Dynamics** Python package

**SINDy** is a machine learning method to identify symbolic differential
equations of dynamical systems from time series data [#]_. The main existing
package is pysindy_, for scalar coordinates.

.. _pysindy: https://github.com/dynamicslab/pysindy

In matSINDy we extend this approach to symbolic variables which are dense,
low-dimension (typically 3 by 3) **matrices**. It works as follows: from a time
series of a matrix $A(t)$ and forcing $B(t)$ (which can also be a matrix),
we compute the time derivative of the series, $\\dot A$, and an arbitrary set of
features $f(A, B)$ (polynomial features for example). The goal is to find a sparse
regression:

$$ \\dot A = \\sum_i \\beta_i f_i(A, B) $$

so that the data is described by the model, and the number of features is small.

For example, in polymer physics, we often describe the evolution of the stress
tensor, $\\tau$, by a *constitutive* equation of the form:

$$ \\dot \\tau = \\sum_i \\beta_i f_i(\\tau, \\nabla U) $$

where $\\nabla U$ is the velocity gradient tensor. The `upper-convected Maxwell model`_
is a concrete example. What other constitutive laws can we find from data [#]?...

.. _`upper-convected Maxwell model`: https://en.wikipedia.org/wiki/Upper-convected_Maxwell_model

Because computing the time derivative of a noisy input amplifies the noise, we prefer
to work with a *weak formulation* of the problem [#]_: the input and features are projected
onto a set of test functions. This is equivalent to linear filtering followed by
downsampling. The complete pipeline is represented in the figure below:

.. image:: docs/data.png
    :align: center
    :alt: Data processing pipeline

Of course, the models should be cross-validated, as shown in the examples notebooks.

⚠️ Work in progress!



License
-------
Copyright (C) 2022 by Etienne Rognin <ecr43 at cam.ac.uk>

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.



Installation
------------
In the target directory, clone this repository::

  git clone https://github.com/etiennerognin/matSINDy.git

Then run the install script::

  pip install -U .


Usage
-----
Note that this package has a different usage from ``pysindy``.

Examples
^^^^^^^^
See the Jupyter notebooks in the ``/examples`` folder.


Related packages
----------------

pysindy_
  Main SINDy package.

.. _pysindy: https://github.com/dynamicslab/pysindy


References
----------

.. [#] S Brunton, J Proctor, and J Kutz, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems* (https://www.pnas.org/doi/10.1073/pnas.1517384113)
.. [#] D Messenger, D Bortz, *Weak SINDy: Galerkin-Based Data-Driven Model Selection* (https://arxiv.org/abs/2005.04339)
.. [#] N Seryo *Learning the constitutive relation of polymeric flows with memory* (https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033107)
