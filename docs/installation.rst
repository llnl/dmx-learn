Installation
============

*dmx-learn* requires Python 3.10 or later. It can be installed from a local
checkout with Poetry or pip. Poetry is recommended for development because it
manages the project's locked dependencies and optional dependency groups.

Using Poetry
------------

Install Poetry first if it is not already available:

.. code-block:: console

   $ curl -sSL https://install.python-poetry.org | python3 -

From the repository checkout, install the core dependencies:

.. code-block:: console

   $ cd /path/to/dmx-learn
   $ poetry install

Install optional features with Poetry extras:

.. code-block:: console

   $ poetry install -E torch       # PyTorch support
   $ poetry install -E optional    # MPI and UMAP support
   $ poetry install -E all         # All optional features

For development or documentation work, install the corresponding dependency
group:

.. code-block:: console

   $ poetry install --with dev     # Testing and development tools
   $ poetry install --with docs    # Documentation-building tools

Using pip
---------

Install the package from a local checkout with pip:

.. code-block:: console

   $ pip install /path/to/dmx-learn

The same optional features are available as pip extras:

.. code-block:: console

   $ pip install /path/to/dmx-learn[torch]
   $ pip install /path/to/dmx-learn[optional]
   $ pip install /path/to/dmx-learn[all]

Optional dependencies
---------------------

The available extras and Poetry groups are summarized below.

.. list-table:: Extras for pip and Poetry
   :header-rows: 1
   :widths: 15 35 25 25

   * - Extra
     - Install command
     - Includes
     - Use case
   * - ``torch``
     - ``pip install .[torch]`` or ``poetry install -E torch``
     - PyTorch
     - GPU-accelerated distributions
   * - ``optional``
     - ``pip install .[optional]`` or ``poetry install -E optional``
     - mpi4py, umap-learn
     - Distributed computing
   * - ``all``
     - ``pip install .[all]`` or ``poetry install -E all``
     - All optional dependencies
     - All optional features

.. list-table:: Poetry groups for development
   :header-rows: 1
   :widths: 15 35 25 25

   * - Group
     - Install command
     - Includes
     - Use case
   * - ``dev``
     - ``poetry install --with dev``
     - pytest, pytest-dependency
     - Testing
   * - ``docs``
     - ``poetry install --with docs``
     - sphinx, sphinx-rtd-theme
     - Documentation building

Extras and groups can be combined:

.. code-block:: console

   $ poetry install -E all --with dev,docs
   $ pip install .[torch,optional]

To inspect packages installed in the Poetry environment, run:

.. code-block:: console

   $ poetry show
