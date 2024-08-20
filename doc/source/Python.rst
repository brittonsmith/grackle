.. _python:

Pygrackle: Running Grackle in Python
====================================

Grackle comes with a Python interface, called Pygrackle, which provides
access to all of Grackle's functionality.  Pygrackle requires the following
Python packages:

 - `Cython <https://cython.org/>`__

 - flake8 (only required for the test suite)

 - `h5py <https://www.h5py.org/>`__

 - `matplotlib <https://matplotlib.org/>`__

 - `NumPy <https://www.numpy.org/>`__

 - packaging (only required for the test suite)

 - py.test (only required for the test suite)

 - `yt <https://yt-project.org/>`__

The easiest thing to do is follow the instructions for installing yt,
which will provide you with Cython, matplotlib, and NumPy.  Flake8 and
py.test can then be installed via pip.

.. _install-pygrackle:

Installing Pygrackle
--------------------

Once the Grackle library has been built and the above dependencies have been
installed, Pygrackle can be installed by moving into the **src/python**
directory and invoking the ``pip install -e .`` command.

This is particularly simple if you installed the Grackle library with the classic build system:

.. code-block:: shell-session

   ~/grackle $ cd src/python
   ~/grackle/src/python $ pip install -e .


If you used the cmake build-system, you also need to store the path to the build directory in the ``PYGRACKLE_CMAKE_BUILD_DIR`` environment variable.
If your build directory was located in the root level of the grackle repository and was called **my-build**, the command would look like:

.. code-block:: shell-session

   ~/grackle $ cd src/python
   ~/grackle/src/python $ PYGRACKLE_CMAKE_BUILD_DIR=../../my-build pip install -e .

.. note:: Pygrackle can only be run when Grackle is compiled without OpenMP.
   See :ref:`openmp`.

.. _pygrackle-dev:

Installing Pygrackle Development Requirements
+++++++++++++++++++++++++++++++++++++++++++++

There are a handful of additional packages required for developing
Grackle. For example, these will enable :ref:`testing` and building
the documentation locally. To install the development dependencies,
repeat the last line of the :ref:`pygrackle installation instructions
<install-pygrackle>` with ``[dev]`` appended.

.. code-block:: shell-session

   ~/grackle/src/python $ pip install -e .[dev]

or

.. code-block:: shell-session

   ~/grackle/src/python $ PYGRACKLE_CMAKE_BUILD_DIR=../../my-build pip install -e .[dev]

If you use ``zsh`` as your shell, you will need quotes around
'.[dev]'.

.. code-block:: shell-session

   ~/grackle/src/python $ pip install -e '.[dev]'


Running the Example Scripts
---------------------------

A number of example scripts are available in the **src/python/examples**
directory.  These scripts provide examples of ways that Grackle can be
used in simplified models, such as solving the temperature evolution of
a parcel of gas at constant density or in a free-fall model.  Each example
will produce a figure as well as a dataset that can be loaded and analyzed
with `yt <http://yt-project.org/>`__.

Configuring the path to Grackle input data
++++++++++++++++++++++++++++++++++++++++++

All of the example scripts discussed below use the following line to
make a guess at where the Grackle input files are located.

.. code-block:: python

   from pygrackle.utilities.data_path import grackle_data_dir

This will typically work for any 'editable' Pygrackle installation
(i.e., one installed with ``pip install -e .`` as directed
above). In this case, it will be assumed that the data files can be
found in a directory called ``input`` in the top level of the source
repository. However, this will not work with non-editable
installations. In this case you can use the ``GRACKLE_DATA_DIR``
environment variable to set the path to the data. This will be picked
up by the Python code above and the ``grackle_data_dir`` variable will
contain the proper path.

.. code-block:: shell-session

   export GRACKLE_DATA_DIR=/path/to/data

Cooling Rate Figure Example
+++++++++++++++++++++++++++

This sets up a one-dimensional grid at a constant density with 
logarithmically spaced temperatures from 10 K to 10\ :sup:`9` K.  Radiative cooling 
is disabled and the chemistry solver is iterated until the species fractions 
have converged.  The cooling time is then calculated and used to compute the cooling 
rate.

.. code-block:: shell-session

   ~/grackle/src/python/examples $ python cooling_rate.py

.. image:: _images/cooling_rate.png
   :width: 500

After the script runs, and hdf5 file will be created with a similar name.  This
can be loaded in with yt.

.. code-block:: python

   >>> import yt
   >>> ds = yt.load("cooling_rate.h5")
   >>> print ds.data["temperature"]
   [  1.00000000e+01   1.09698580e+01   1.20337784e+01   1.32008840e+01, ...,
      7.57525026e+08   8.30994195e+08   9.11588830e+08   1.00000000e+09] K
   >>> print ds.data["cooling_rate"]
   [  1.09233398e-25   1.08692516e-25   1.08117583e-25   1.07505345e-25, ...,
      3.77902570e-23   3.94523273e-23   4.12003667e-23   4.30376998e-23] cm**3*erg/s


Cooling Cell Example
++++++++++++++++++++

This sets up a single grid cell with an initial density and temperature and solves 
the chemistry and cooling for a given amount of time.  The resulting dataset gives
the values of the densities, temperatures, and mean molecular weights for all times.

.. code-block:: shell-session

   ~/grackle/src/python/examples $ python cooling_cell.py

.. image:: _images/cooling_cell.png
   :width: 500

.. code-block:: python

   >>> import yt
   >>> ds = yt.load("cooling_cell.h5")
   >>> print ds.data["time"].to("Myr")
   YTArray([  0.00000000e+00,   6.74660169e-02,   1.34932034e-01, ...,
            9.98497051e+01,   9.99171711e+01,   9.99846371e+01]) Myr
   >>> print ds.data["temperature"]
   YTArray([ 990014.56406726,  980007.32720091,  969992.99066987, ...,
             9263.81515866,    9263.81515824,    9263.81515865]) K


Free-Fall Collapse Example
++++++++++++++++++++++++++

This sets up a single grid cell with an initial number density of 1 cm\ :sup:`-3`.  
The density increases with time following a free-fall collapse model.  As the density 
increases, thermal energy is added to model heating via adiabatic compression.
This can be useful for testing chemistry networks over a large range in density.

.. code-block:: shell-session

   ~/grackle/src/python/examples $ python freefall.py

.. image:: _images/freefall.png
   :width: 500

The resulting dataset can be analyzed similarly as above.

.. code-block:: python

   >>> import yt
   >>> ds = yt.load("freefall.h5")
   >>> print ds.data["time"].to("Myr")
   [   0.            0.45900816    0.91572127 ...,  219.90360841  219.90360855
     219.9036087 ] Myr
   >>> print ds.data["density"]
   [  1.67373522e-25   1.69059895e-25   1.70763258e-25 ...,   1.65068531e-12
      1.66121253e-12   1.67178981e-12] g/cm**3
   >>> print ds.data["temperature"]
   [   99.94958248   100.61345564   101.28160228 ...,  1728.89321898
     1729.32604568  1729.75744287] K

Using Grackle with yt
+++++++++++++++++++++

This example illustrates how Grackle functionality can be called using
simulation datasets loaded with `yt <https://yt-project.org/>`__ as
input. Note, below we invoke Python with the ``-i`` flag to keep the
interpreter running. The second block is assumed to happen within the
same session.

.. code-block:: shell-session

   ~/grackle/src/python/examples $ python -i yt_grackle.py

.. code-block:: python

   >>> print (sp['gas', 'grackle_cooling_time'].to('Myr'))
   [-5.33399975 -5.68132287 -6.04043746 ... -0.44279721 -0.37466095
    -0.19981158] Myr
   >>> print (sp['gas', 'grackle_temperature'])
   [12937.90890302 12953.99126155 13234.96820101 ... 11824.51319307
    11588.16161462 10173.0168747 ] K

Through ``pygrackle``, the following ``yt`` fields are defined:

- ``('gas', 'grackle_cooling_time')``
- ``('gas', 'grackle_gamma')``
- ``('gas', 'grackle_molecular_weight')``
- ``('gas', 'grackle_pressure')``
- ``('gas', 'grackle_temperature')``
- ``('gas', 'grackle_dust_temperature')``

These fields are created after calling the ``add_grackle_fields`` function.
This function will initialize Grackle with settings from parameters in the
loaded dataset. Optionally, parameters can be specified manually to override.
