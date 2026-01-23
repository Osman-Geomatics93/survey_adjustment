# Contributing to Survey Adjustment & Network Analysis

First off, thank you for considering contributing to this project! Your help makes this plugin better for the entire surveying community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

---

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**Great bug reports include:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Input CSV files (anonymized if needed)
- Output JSON/HTML report
- QGIS version and plugin version
- Operating system

### Suggesting Features

Feature suggestions are welcome! Please include:
- Clear description of the feature
- Use case / why it's needed
- Any relevant references or examples

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes
6. Push to your fork
7. Open a Pull Request

---

## Development Setup

### Prerequisites

- Python 3.9+
- QGIS 3.22+ (for integration testing)
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/survey_adjustment.git
cd survey_adjustment

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install pytest numpy

# Run tests to verify setup
pytest
```

### Running in QGIS

1. Create a symbolic link from your development folder to QGIS plugins directory:
   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

2. Enable the plugin in QGIS Plugin Manager
3. Use the Plugin Reloader plugin for quick iteration

---

## Project Structure

```
survey_adjustment/
├── core/                    # QGIS-independent computation (pure Python)
│   ├── models/              # Data classes (Point, Network, Observation)
│   ├── solver/              # Least squares algorithms
│   ├── statistics/          # Statistical tests
│   ├── geometry/            # Error ellipses, vectors
│   ├── results/             # Result structures
│   ├── reports/             # Report generation
│   └── validation/          # Constraint health checks
├── qgis_integration/        # QGIS-specific code
│   ├── algorithms/          # Processing algorithms
│   ├── gui/                 # Dialogs and UI
│   ├── io/                  # CSV/layer parsing
│   ├── plugin.py            # Plugin lifecycle
│   └── provider.py          # Processing provider
├── tests/                   # Test suite
│   ├── data/                # Test datasets
│   └── test_*.py            # Test files
└── examples/                # Example datasets
```

### Key Design Principles

1. **Core is QGIS-free**: All computation in `core/` must work without QGIS imports
2. **No SciPy dependency**: Statistics implemented from scratch for portability
3. **Lazy imports**: QGIS code only imported when needed
4. **Type hints**: Use type annotations for better code quality

---

## Coding Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Add docstrings to public functions/classes
- Keep functions focused and small

### Example

```python
def compute_azimuth(from_point: Point, to_point: Point) -> float:
    """Compute azimuth from North, clockwise positive.

    Args:
        from_point: Origin point
        to_point: Target point

    Returns:
        Azimuth in radians [0, 2π)
    """
    dE = to_point.easting - from_point.easting
    dN = to_point.northing - from_point.northing
    return math.atan2(dE, dN) % (2 * math.pi)
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add Danish robust estimation method

- Implement Danish weight function with configurable c parameter
- Add unit tests for edge cases
- Update documentation
```

Prefixes:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code refactoring
- `style:` Formatting changes

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_solver_2d.py

# Run with coverage
pytest --cov=survey_adjustment

# Run verbose
pytest -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include edge cases

```python
def test_leveling_adjustment_converges():
    """Test that leveling adjustment converges for valid input."""
    network = create_test_leveling_network()
    result = adjust_leveling_1d(network)

    assert result.success
    assert result.converged
    assert result.iterations < 10
```

---

## Submitting Changes

### Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Create Pull Request with clear description

### PR Template

Your PR description should include:
- **What:** Brief description of changes
- **Why:** Motivation for the change
- **How:** Technical approach taken
- **Testing:** How you tested the changes

---

## Questions?

Feel free to:
- Open an issue for discussion
- Contact the maintainer

Thank you for contributing!
