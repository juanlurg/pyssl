Contributing to PySSL
====================

We welcome contributions to PySSL! This guide will help you get started, whether you want to report a bug, request a feature, or contribute code.

🎯 Ways to Contribute
--------------------

There are many ways to contribute to PySSL:

* **🐛 Report bugs** - Help us identify and fix issues
* **💡 Request features** - Suggest new functionality
* **📝 Improve documentation** - Help make PySSL more accessible
* **🔧 Submit code** - Fix bugs or implement new features
* **🧪 Add tests** - Improve test coverage
* **📊 Share examples** - Show PySSL in action

🚀 Quick Start for Contributors
------------------------------

### 1. Set Up Development Environment

.. code-block:: bash

   # Fork the repository on GitHub first
   git clone https://github.com/YOUR-USERNAME/pyssl.git
   cd pyssl

   # Install development dependencies
   pip install -e ".[test,docs]"

   # Run tests to verify setup
   pytest tests/

### 2. Create a Branch

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description

### 3. Make Your Changes

* Write clear, well-documented code
* Add tests for new functionality
* Update documentation as needed

### 4. Submit Your Contribution

.. code-block:: bash

   # Run tests and linting
   pytest tests/

   # Commit your changes
   git add .
   git commit -m "Add feature: brief description"

   # Push to your fork
   git push origin feature/your-feature-name

   # Create a pull request on GitHub

🐛 Reporting Bugs
----------------

Before reporting a bug, please:

1. **Check existing issues** - Your bug might already be reported
2. **Update to latest version** - The bug might already be fixed
3. **Create a minimal example** - Help us reproduce the issue

### Bug Report Template

When reporting bugs, please include:

.. code-block:: markdown

   **Bug Description**
   A clear description of what the bug is.

   **To Reproduce**
   ```python
   # Minimal code example that reproduces the bug
   import ssl_framework
   # ... your code here
   ```

   **Expected Behavior**
   What you expected to happen.

   **Actual Behavior**
   What actually happened.

   **Environment**
   - PySSL version: [e.g., 0.1.0]
   - Python version: [e.g., 3.9.0]
   - Operating System: [e.g., Ubuntu 20.04]
   - scikit-learn version: [e.g., 1.3.0]

   **Additional Context**
   Any other relevant information.

💡 Requesting Features
---------------------

We love feature requests! Please:

1. **Check existing requests** - Your idea might already be discussed
2. **Explain the use case** - Help us understand why it's needed
3. **Suggest implementation** - If you have ideas about how to implement it

### Feature Request Template

.. code-block:: markdown

   **Feature Description**
   A clear description of the feature you'd like to see.

   **Use Case**
   Describe the problem this feature would solve.

   **Proposed Solution**
   How you envision this feature working.

   **Alternatives**
   Any alternative solutions you've considered.

   **Additional Context**
   Any other relevant information.

🔧 Development Guidelines
------------------------

### Code Style

We follow these conventions:

* **PEP 8** for Python code style
* **Type hints** for all public functions
* **Docstrings** for all public classes and methods
* **Clear variable names** - prefer descriptive over concise

Example:

.. code-block:: python

   def select_confident_samples(
       X_unlabeled: np.ndarray,
       y_proba: np.ndarray,
       threshold: float = 0.95
   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
       """Select samples above confidence threshold.

       Parameters
       ----------
       X_unlabeled : np.ndarray
           Unlabeled feature data.
       y_proba : np.ndarray
           Predicted probabilities.
       threshold : float, default=0.95
           Confidence threshold.

       Returns
       -------
       X_selected : np.ndarray
           Selected feature data.
       y_selected : np.ndarray
           Selected pseudo-labels.
       indices : np.ndarray
           Selected sample indices.
       """
       # Implementation here
       pass

### Testing

All code contributions must include tests:

* **Unit tests** for individual functions
* **Integration tests** for complete workflows
* **Edge case tests** for boundary conditions

Example test:

.. code-block:: python

   def test_confidence_threshold_selection():
       """Test that ConfidenceThreshold selects correct samples."""
       # Arrange
       X_unlabeled = np.array([[1, 2], [3, 4], [5, 6]])
       y_proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.98, 0.02]])
       strategy = ConfidenceThreshold(threshold=0.95)

       # Act
       X_selected, y_selected, indices = strategy.select_labels(X_unlabeled, y_proba)

       # Assert
       assert len(X_selected) == 2  # Samples 0 and 2 should be selected
       np.testing.assert_array_equal(indices, [0, 2])

### Documentation

Update documentation for:

* **New features** - Add to user guide and API reference
* **API changes** - Update docstrings and examples
* **Bug fixes** - Note in changelog

Documentation is written in reStructuredText and built with Sphinx.

📁 Project Structure
-------------------

Understanding the project layout:

.. code-block:: text

   pyssl/
   ├── ssl_framework/           # Main package
   │   ├── __init__.py
   │   ├── main.py             # SelfTrainingClassifier
   │   └── strategies.py       # Selection/Integration strategies
   ├── tests/                  # Test suite
   │   ├── test_main.py
   │   └── test_strategies.py
   ├── docs/                   # Documentation
   │   └── source/
   ├── examples/               # Example scripts and notebooks
   ├── pyproject.toml          # Project configuration
   └── README.md

🧪 Running Tests
---------------

### Basic Testing

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test file
   pytest tests/test_main.py

   # Run with coverage
   pytest tests/ --cov=ssl_framework --cov-report=html

### Test Types

* **Unit tests** - Test individual functions/methods
* **Integration tests** - Test complete workflows
* **Strategy tests** - Test selection/integration strategies

### Writing Good Tests

1. **Test the interface, not implementation**
2. **Use descriptive test names**
3. **Follow Arrange-Act-Assert pattern**
4. **Test edge cases and error conditions**

📚 Documentation
---------------

### Building Documentation

.. code-block:: bash

   # Build HTML documentation
   sphinx-build -b html docs/source docs/build

   # Serve locally
   python -m http.server -d docs/build 8000

### Documentation Types

* **API Reference** - Auto-generated from docstrings
* **User Guide** - Tutorials and how-to guides
* **Examples** - Jupyter notebooks and scripts

🔄 Pull Request Process
----------------------

### Before Submitting

1. **✅ Tests pass** - All existing and new tests
2. **✅ Documentation updated** - For new features
3. **✅ Code style** - Follows project conventions
4. **✅ Clear commit messages** - Describe what and why

### PR Review Process

1. **Automated checks** - Tests, linting, coverage
2. **Code review** - Maintainer review for correctness
3. **Documentation review** - Clarity and completeness
4. **Final approval** - Merge when ready

### PR Guidelines

* **Clear title** - Summarize the change
* **Detailed description** - Explain what and why
* **Link issues** - Reference related issues
* **Small focused changes** - Easier to review

Example PR description:

.. code-block:: markdown

   ## Summary
   Adds support for custom confidence thresholds in TopKFixedCount strategy.

   ## Changes
   - Add optional `min_confidence` parameter to TopKFixedCount
   - Update tests to cover new functionality
   - Add documentation example

   ## Motivation
   Addresses issue #123 where users wanted to combine TopK selection with minimum confidence requirements.

   ## Testing
   - Added unit tests for new parameter
   - Verified existing tests still pass
   - Tested with real dataset in examples/

🏷️ Coding Standards
------------------

### Python Standards

* **Python 3.8+** compatibility
* **Type hints** for public APIs
* **Docstrings** following NumPy style
* **Error handling** with informative messages

### API Design

* **Scikit-learn compatibility** - Follow sklearn conventions
* **Modular design** - Clear separation of concerns
* **Backward compatibility** - Avoid breaking changes
* **Clear interfaces** - Well-defined strategy protocols

### Performance

* **Efficient NumPy operations** - Vectorized computations
* **Memory conscious** - Handle large datasets appropriately
* **Benchmark critical paths** - Measure performance impact

🌟 Recognition
-------------

All contributors are recognized in:

* **CONTRIBUTORS.md** - List of all contributors
* **Release notes** - Credit for specific contributions
* **Documentation** - Author attribution where appropriate

🤝 Community Guidelines
----------------------

We strive to maintain a welcoming, inclusive community:

* **Be respectful** - Treat everyone with kindness
* **Be constructive** - Provide helpful feedback
* **Be patient** - Everyone is learning
* **Assume good intent** - Give others benefit of doubt

### Getting Help

If you need help:

* **📖 Read the docs** - Most questions are answered here
* **🔍 Search issues** - Someone might have asked before
* **💬 Start a discussion** - Use GitHub Discussions for questions
* **📧 Contact maintainers** - For sensitive issues

🚀 Advanced Contributions
------------------------

### New Strategy Types

Want to implement a new selection or integration strategy?

1. **Study existing strategies** - Understand the interface
2. **Implement the protocol** - Follow method signatures
3. **Add comprehensive tests** - Cover edge cases
4. **Document thoroughly** - Include examples

### Performance Improvements

* **Profile first** - Identify actual bottlenecks
* **Benchmark changes** - Measure improvement
* **Maintain compatibility** - Don't break existing code

### New Features

* **Discuss first** - Open an issue to discuss design
* **Start small** - Implement minimal viable version
* **Iterate** - Refine based on feedback

Thank you for contributing to PySSL! 🎉