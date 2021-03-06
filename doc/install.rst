Installation
============

There are two ways to install `tensorwaves`. Even though the PyPI installation
is the fastest, we recommend following the :ref:`install:Development mode`
procedure.


Installation through PyPI
-------------------------

`tensorwaves` is `available as a PyPI package
<https://pypi.org/project/tensorwaves/>`_, so installation is super easy:

.. code-block:: shell

  pip install tensorwaves

Et voilà, that's it! You can try out whether the installation works by running:

.. code-block:: python

  import tensorwaves

from the `Python interpreter
<https://docs.python.org/3/tutorial/interpreter.html>`_. Note that PyPI only
allows you to install specific releases, so we recommend using the more
dynamic, :ref:`'development mode' <install:Development mode>` instead.


Development mode
----------------

`tensorwaves` is an academic research project and is bound to continuously
evolve. We therefore highly recommend installing `tensorwaves` from `the source
code <https://github.com/ComPWA/tensorwaves>`_, so that you work with the
latest version.

Moreover, since you read as far as this, you must have an interest in particle
physics, and it is researchers like you who can help bring this project
further! So please, follow the following sections to set up this 'interactive
installation'.


.. _local-repository:

Step 1: Get the source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `tensorwaves` source code is maintained through `Git
<https://git-scm.com/>`_, so you need to `install Git
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ first. Once
you've done so, navigate to a suitable folder and run:

.. code-block:: shell

  git clone https://github.com/ComPWA/tensorwaves.git
  cd tensorwaves

After that, there should be a folder called :file:`tensorwaves` into which we
navigated just now. We'll call this folder the **local repository**.


Step 2: Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is safest to install `tensorwaves` within a virtual environment, so that all
Python dependencies are contained within there. This is helpful in case
something goes wrong with the dependencies: you can just trash the environment
and recreate it. There are two options: :ref:`Conda <install:Conda
environment>` or :ref:`Python's venv <install:Python venv>`.

Conda environment
~~~~~~~~~~~~~~~~~

`Conda <https://www.anaconda.com/>`_ can be installed without administrator
rights, see instructions on `this page
<https://www.anaconda.com/distribution/>`_. Once installed, navigate to the
:ref:`local repository <local-repository>` and create the Conda environment for
`tensorwaves` as follows:

.. code-block:: shell

  conda env create

This command uses the `environment.yml
<https://github.com/ComPWA/tensorwaves/blob/master/environment.yml>`_ file and
immediately installs `tensorwaves` in `development mode
<https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__.

After Conda finishes creating the environment, you can activate it with as
follows:

.. code-block:: shell

  conda activate tw


You need to have the :code:`tw` environment activated whenever you want to run
`tensorwaves`.

Python venv
~~~~~~~~~~~

Alternatively, you can use `Python's venv
<https://docs.python.org/3/library/venv.html>`_, if you have that available on
your system. All you have to do, is navigate into :ref:`local repository
<local-repository>` and run:

.. code-block:: shell

  python3 -m venv ./venv

This creates a folder called :file:`venv` where all Python packages will be
contained. You first have to activate the environment, and will have to do so
whenever you want to run `tensorwaves`.

.. code-block:: shell

  source ./venv/bin/activate

Now you can safely install `tensorwaves` in `development mode
<https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`__:

.. code-block:: shell

  pip install -e .

That's it, now you're all set to :doc:`use tensorwaves <usage>`!


Step 3: Test the installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, navigate out of the main directory of the :ref:`local repository
<local-repository>` in order to make sure that `tensorwaves` we run, is the
system installation and not the :file:`tensorwaves` folder in the current
working directory. Then, simply launch a Python interpreter and run:

.. code-block:: python

  import tensorwaves

If you don't get any error messages, all worked out nicely!

For more thorough testing, navigate back to the you can run the unit tests:

.. code-block:: shell

  pip install -e .[test]  # install dependencies for testing
  pytest -m "not slow"

After that, it's worth having a look at the :doc:`contribute page
<contribute>`!

Updating to the latest version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When new commits are merged into the `master branch
<https://github.com/ComPWA/tensorwaves/tree/master>`_, you need to update your
:ref:`local copy of the source code <local-repository>` as follows:

.. code-block:: shell

  git checkout master
  git pull
  pip install -e .

It's best to have a clean your `working tree
<https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>`_
before you do a :command:`git pull`. We also call :command:`pip install` again,
because we sometimes introduce upgrades of the dependencies.

If you face any issues when calling :code:`pip install -e .`, just trash your
:ref:`install:Conda environment` or :ref:`venv <install:Python venv>` and
repeat from :ref:`Step 2 <install:Step 2: Create a virtual environment>`.
