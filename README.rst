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
we compute $\partial A/\partial t$

>> Work in progress!



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

.. [#] S Brunton, J Proctor, and J Kutz *Discovering governing equations from data by sparse identification of nonlinear dynamical systems* (https://www.pnas.org/doi/10.1073/pnas.1517384113)
