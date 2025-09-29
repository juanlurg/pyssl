API Reference
=============

This section provides detailed documentation for all PySSL classes, functions, and modules.

📋 Quick Reference
------------------

**Core Classes:**

* :class:`ssl_framework.main.SelfTrainingClassifier` - Main SSL classifier
* :class:`ssl_framework.strategies.ConfidenceThreshold` - Confidence-based selection
* :class:`ssl_framework.strategies.TopKFixedCount` - Top-K selection
* :class:`ssl_framework.strategies.AppendAndGrow` - Simple integration
* :class:`ssl_framework.strategies.ConfidenceWeighting` - Weighted integration

**Quick Import:**

.. code-block:: python

   from ssl_framework.main import SelfTrainingClassifier
   from ssl_framework.strategies import (
       ConfidenceThreshold, TopKFixedCount,
       AppendAndGrow, FullReLabeling, ConfidenceWeighting
   )

📖 Detailed Documentation
-------------------------

.. toctree::
   :maxdepth: 2

   main
   strategies

🔗 External References
----------------------

PySSL builds on top of these excellent libraries:

* :doc:`scikit-learn <sklearn:index>` - Base estimator interface
* :doc:`numpy <numpy:index>` - Numerical operations
* :doc:`pandas <pandas:index>` - DataFrame support

📝 Type Annotations
------------------

PySSL includes comprehensive type hints. For complete type information, see the source code or use your IDE's type checking capabilities.

.. code-block:: python

   from typing import Union, Optional
   import numpy as np
   import pandas as pd

   # Supported input types
   ArrayLike = Union[np.ndarray, pd.DataFrame]
   TargetLike = Union[np.ndarray, pd.Series]