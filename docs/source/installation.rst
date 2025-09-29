Installation Guide
==================

PySSL can be installed in several ways depending on your needs. This guide covers all installation methods from basic usage to development setup.

📦 Quick Installation
---------------------

### From PyPI (Coming Soon)

Once PySSL is released on PyPI, installation will be as simple as:

.. code-block:: bash

   pip install pyssl

### From Source (Current Method)

For now, install directly from the source repository:

.. code-block:: bash

   git clone https://github.com/yourusername/pyssl.git
   cd pyssl
   pip install -e .

🔧 Development Installation
---------------------------

If you want to contribute to PySSL or run the tests:

.. code-block:: bash

   git clone https://github.com/yourusername/pyssl.git
   cd pyssl
   pip install -e ".[test]"

This installs PySSL in editable mode with all testing dependencies.

📋 Requirements
---------------

PySSL requires Python 3.8 or higher and the following dependencies:

**Core Dependencies:**

* :doc:`numpy <numpy:index>` - Numerical computing
* :doc:`pandas <pandas:index>` - Data manipulation and analysis
* :doc:`scikit-learn <sklearn:index>` - Machine learning library

**Testing Dependencies (for development):**

* :doc:`pytest <pytest:index>` - Testing framework
* :doc:`pytest-cov <pytest-cov:index>` - Coverage reporting

**Documentation Dependencies (for docs):**

* :doc:`sphinx <sphinx:index>` - Documentation generator
* ``pydata-sphinx-theme`` - Modern documentation theme
* ``myst-parser`` - Markdown support for Sphinx

🚀 Verify Installation
----------------------

To verify that PySSL is installed correctly, run this simple test:

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LogisticRegression
   from ssl_framework.main import SelfTrainingClassifier

   # Create sample data
   X_labeled = np.array([[1, 2], [3, 4]])
   y_labeled = np.array([0, 1])
   X_unlabeled = np.array([[2, 3]])

   # Test basic functionality
   ssl_clf = SelfTrainingClassifier(LogisticRegression())
   ssl_clf.fit(X_labeled, y_labeled, X_unlabeled)

   print("✅ PySSL installed successfully!")

If this runs without errors, you're ready to use PySSL!

🐍 Using uv (Recommended for Development)
-----------------------------------------

For the fastest and most reliable development setup, we recommend using `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone and set up PySSL
   git clone https://github.com/yourusername/pyssl.git
   cd pyssl

   # Create virtual environment and install dependencies
   uv sync

   # Run tests
   uv run pytest tests/

🐛 Troubleshooting
------------------

**ImportError: No module named 'ssl_framework'**

Make sure you installed PySSL in editable mode with ``pip install -e .`` and that you're in the correct directory.

**ModuleNotFoundError: No module named 'sklearn'**

Install scikit-learn:

.. code-block:: bash

   pip install scikit-learn

**Tests failing during development setup**

Ensure all test dependencies are installed:

.. code-block:: bash

   pip install -e ".[test]"
   pytest tests/

**Documentation build failing**

Install documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build

🔄 Updating PySSL
-----------------

To update your PySSL installation:

**From source:**

.. code-block:: bash

   cd pyssl
   git pull origin main
   pip install -e .

**From PyPI (when available):**

.. code-block:: bash

   pip install --upgrade pyssl

📚 Next Steps
-------------

Now that you have PySSL installed, check out:

* :doc:`getting_started` - Learn the basics
* :doc:`quickstart_tutorial` - 5-minute tutorial
* :doc:`examples/basic_usage` - Complete examples