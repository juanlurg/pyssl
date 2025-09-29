# SSL Framework Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating the semi-supervised learning framework.

## 📚 Notebook Overview

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| `01_quickstart.ipynb` | 5-minute demo showing SSL value | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pyssl/blob/main/notebooks/01_quickstart.ipynb) |
| `02_classification_comparison.ipynb` | Compare different SSL strategies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pyssl/blob/main/notebooks/02_classification_comparison.ipynb) |
| `04_tabular_data_pipeline.ipynb` | Production ML pipelines with SSL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pyssl/blob/main/notebooks/04_tabular_data_pipeline.ipynb) |
| `05_hyperparameter_tuning.ipynb` | Optimize SSL parameters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pyssl/blob/main/notebooks/05_hyperparameter_tuning.ipynb) |
| `06_production_patterns.ipynb` | Enterprise deployment patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pyssl/blob/main/notebooks/06_production_patterns.ipynb) |

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for experimentation)
1. Click any "Open in Colab" button above
2. Run the first setup cell to install dependencies
3. Follow along with the examples

### Option 2: Local Jupyter
1. Clone this repository
2. Install dependencies: `pip install -e .`
3. Start Jupyter: `jupyter lab notebooks/`
4. Open any notebook and run the cells

## 🔧 Setup Notes

### For Repository Maintainers
Before publishing, update all GitHub URLs:
1. Replace `yourusername/pyssl` with your actual GitHub repository
2. Update the URLs in:
   - All Colab badges in notebook cells
   - This README file
   - Any setup cells that reference the repository

### Google Colab Compatibility
Each notebook includes a setup cell that:
- Detects if running in Google Colab
- Installs required dependencies automatically
- Provides fallback framework definitions for compatibility

## 📊 Dataset Utilities

The `utils/data_generation.py` module provides:
- `generate_ssl_dataset()` - Standard SSL data splits
- `make_imbalanced_classification()` - Imbalanced datasets
- `create_ssl_benchmark()` - Predefined benchmark scenarios

## 🎯 Learning Path

**Recommended order:**
1. **Quickstart** → Understand SSL value proposition
2. **Classification Comparison** → Learn different strategies
3. **Tabular Pipeline** → See production integration
4. **Hyperparameter Tuning** → Optimize performance
5. **Production Patterns** → Deploy at scale

## 🐛 Troubleshooting

**Common Issues:**
- **Import errors in Colab**: Run the setup cell at the top of each notebook
- **Missing dependencies**: Install with `pip install matplotlib seaborn scikit-learn`
- **Framework not found**: Ensure you're running from the correct directory

**Need help?** Open an issue in the main repository.