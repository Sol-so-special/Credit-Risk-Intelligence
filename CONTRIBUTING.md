# Contributing Guidelines

Contributions to the **Credit Risk Intelligence ML Pipeline** are welcome! This document provides guidelines for contributing to the project.

___

## 🤝 How to Contribute

### Types of Contributions

We accept the following types of contributions:
- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📚 Documentation improvements
- 🧪 Test coverage improvements
- 🎨 UI/UX improvements
- ⚡ Performance optimizations

___

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with pandas, scikit-learn, and Streamlit

### Development Setup

1. **Fork the repository**
```bash
# Click the "Fork" button on GitHub, then clone your fork
git clone https://github.com/Sol-so-special/Credit-Risk-Intelligence
cd Credit-Risk-Intelligence
```

2. **Set up development environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with development tools
pip install -r requirements.txt
pip install pytest black flake8 pre-commit
```

3. **Install pre-commit hooks**
```bash
pre-commit install
```

4. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

___

## 📝 Development Guidelines

### Code Style

- **Python Style**: Follow PEP 8 guidelines
- **Formatting**: Use `black` for code formatting
- **Linting**: Use `flake8` for code linting
- **Documentation**: Use Google-style docstrings

```python
def calculate_risk_probability(customer_data: dict) -> float:
    """Calculate the probability of credit default for a customer.
    
    Args:
        customer_data (dict): Dictionary containing customer features
            like age, job, balance, etc.
    
    Returns:
        float: Probability of default (0.0 to 1.0)
    
    Raises:
        ValueError: If required features are missing
    
    Example:
        >>> customer = {'age': 35, 'job': 'management', 'balance': 1500}
        >>> prob = calculate_risk_probability(customer)
        >>> print(f"Default probability: {prob:.2%}")
    """
    pass
```

### Project Structure Guidelines

- **utils/**: Core utility modules
  - `data_processor.py`: Data cleaning and validation
  - `model_loader.py`: ML model training and inference
  - `visualizations.py`: Plotting and chart generation
- **tests/**: Unit and integration tests
- **docs/**: Additional documentation
- **notebooks/**: Jupyter notebooks for experimentation

### Commit Message Format

Use conventional commit format:
```
type(scope): brief description

Detailed explanation if needed

- Bullet points for multiple changes
- Reference issues like "Fixes #123"
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```bash
feat(model): add support for gradient boosting models
fix(data): handle missing values in education column
docs(readme): update installation instructions
test(utils): add unit tests for data validation
```

___

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_data_processor.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_data_processor.py
import pytest
import pandas as pd
from utils.data_processor import clean_dataset, validate_input_data

def test_clean_dataset_removes_marketing_columns():
    """Test that marketing columns are properly removed."""
    # Arrange
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'job': ['admin', 'management', 'services'],
        'default': ['no', 'yes', 'no'],
        'duration': [120, 180, 90],  # Marketing column
        'campaign': [1, 2, 1]  # Marketing column
    })
    
    # Act
    df_clean = clean_dataset(df)
    
    # Assert
    assert 'duration' not in df_clean.columns
    assert 'campaign' not in df_clean.columns
    assert 'age' in df_clean.columns
    assert len(df_clean) == 3

def test_validate_input_data_with_missing_features():
    """Test input validation with missing required features."""
    # Arrange
    input_data = {'age': 30}  # Missing other required features
    feature_names = ['age', 'job', 'balance']
    
    # Act
    errors = validate_input_data(input_data, feature_names)
    
    # Assert
    assert len(errors) == 2
    assert 'Missing value for job' in errors
    assert 'Missing value for balance' in errors
```

___

## 📋 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
```bash
pytest
black .
flake8 .
```

2. **Update documentation** if needed
3. **Add tests** for new features
4. **Update CHANGELOG.md** if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts

## Related Issues
Closes #123
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: Maintainers review code for quality and correctness
3. **Testing**: Manual testing for UI changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge preferred

___

## 🎯 Feature Requests

### Proposing New Features

1. **Check existing issues** to avoid duplicates
2. **Create an issue** with detailed description
3. **Include use cases** and business justification
4. **Provide mockups** for UI changes
5. **Discuss implementation** approach

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Business Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches were considered?

## Additional Context
Screenshots, mockups, or related examples.
```

___

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 10, macOS 11.6, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96.0] (for Streamlit issues)

## Screenshots
If applicable, add screenshots to help explain the problem.

## Additional Context
Any other context about the problem.
```

___

## 📚 Documentation

### Documentation Standards

- **README.md**: Keep updated with new features
- **Code Comments**: Explain complex business logic
- **Docstrings**: Document all public functions and classes
- **Type Hints**: Use type hints for better code clarity

### Building Documentation

```bash
# Generate API documentation (if using Sphinx)
cd docs/
make html

# Preview documentation
python -m http.server 8000 -d docs/_build/html
```

___

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in appropriate files
- [ ] Tag created: `git tag v1.2.0`
- [ ] Release notes prepared

___

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support
- **Pull Requests**: Code contributions and reviews

### Code of Conduct

- **Be respectful** and inclusive
- **Provide constructive feedback**
- **Focus on the code**, not the person
- **Help newcomers** get started
- **Follow project guidelines**

___

## 📊 Performance Guidelines

### ML Model Contributions

- **Benchmark against existing models** using ROC-AUC
- **Include cross-validation results**
- **Document hyperparameter choices**
- **Provide feature importance analysis**
- **Test with different data sizes**

### Code Performance

- **Profile critical paths** using cProfile
- **Optimize data processing** with vectorized operations
- **Cache expensive computations** appropriately
- **Monitor memory usage** for large datasets

___

## 🔧 Development Tools

### Recommended IDE Setup

**Visual Studio Code**:
```json
{
    "python.defaultInterpreter": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

### Useful Commands

```bash
# Format code
black . --line-length 88

# Lint code
flake8 . --max-line-length=88 --ignore=E203,W503

# Run tests with coverage
pytest --cov=utils --cov-report=term-missing

# Profile code
python -m cProfile -o profile.stats your_script.py

# Check dependencies
pip-audit  # Security vulnerabilities
pipdeptree  # Dependency tree
```

___

## 🎉 Recognition

Contributors will be recognized in:
- **GitHub releases** notes
- **Annual contributor appreciation** posts

Thank you for contributing to the Credit Risk Intelligence ML Pipeline! Your efforts help make financial technology more accessible and reliable.

---

**Questions?** Feel free to create an issue or start a discussion!