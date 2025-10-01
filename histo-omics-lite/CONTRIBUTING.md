# Contributing to Histo-Omics-Lite

We love your input! We want to make contributing to Histo-Omics-Lite as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Request Process

1. **Fork the repo** and create your branch from `main`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Set up pre-commit hooks**: `pre-commit install`
4. **Make your changes** with appropriate tests
5. **Run quality checks**: `make quality`
6. **Update documentation** if you changed APIs
7. **Ensure tests pass** and maintain >90% coverage
8. **Submit your pull request**

### Branch Naming

Use descriptive branch names:
- `feature/add-new-encoder` - for new features
- `fix/memory-leak-in-training` - for bug fixes
- `docs/improve-quickstart` - for documentation
- `refactor/simplify-config-loading` - for refactoring

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- 8GB+ RAM recommended

### Quick Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/histo-omics-lite.git
cd histo-omics-lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
make test
```

### Development Workflow

```bash
# Run all quality checks
make quality

# Individual checks
make lint          # Ruff linting and formatting
make typecheck     # MyPy type checking
make test          # Tests with coverage
make docs-serve    # Local documentation server

# Before committing
make clean         # Clean build artifacts
make quality       # Final quality check
```

## Code Style

### Python Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for both linting and formatting:

```bash
# Format code
ruff format .

# Lint code  
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Hints

- **Use type hints everywhere** - we run mypy in strict mode
- **Import from `__future__`** for Python 3.10 compatibility:
  ```python
  from __future__ import annotations
  ```
- **Use modern typing syntax** when possible:
  ```python
  # Good
  def process_data(items: list[str]) -> dict[str, int]:
      ...
  
  # Avoid (but OK for compatibility)
  from typing import List, Dict
  def process_data(items: List[str]) -> Dict[str, int]:
      ...
  ```

### Documentation

- **Use Google-style docstrings**:
  ```python
  def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
      """Compute evaluation metrics.
      
      Args:
          predictions: Model predictions [N, C] or [N,]
          targets: Ground truth targets [N,] 
          
      Returns:
          Dictionary containing computed metrics
          
      Raises:
          ValueError: If input tensors have incompatible shapes
      """
  ```

- **Document all public APIs**
- **Include examples in docstrings** for complex functions
- **Keep docstrings up to date** with code changes

## Testing

### Writing Tests

- **Write tests for all new functionality**
- **Use descriptive test names**: `test_compute_retrieval_metrics_perfect_alignment`
- **Follow the AAA pattern**: Arrange, Act, Assert
- **Use fixtures** for common test data (see `tests/conftest.py`)
- **Mark slow tests**: `@pytest.mark.slow`
- **Mock external dependencies** when needed

### Test Structure

```python
class TestMyFeature:
    """Test my new feature."""
    
    def test_basic_functionality(self) -> None:
        """Test basic functionality works correctly."""
        # Arrange
        input_data = create_test_data()
        
        # Act  
        result = my_function(input_data)
        
        # Assert
        assert result.shape == (10, 5)
        assert torch.all(result >= 0)
    
    def test_edge_case_empty_input(self) -> None:
        """Test handling of empty input."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            my_function([])
```

### Running Tests

```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=histo_omics_lite --cov-report=html

# Specific test file
pytest tests/unit/test_evaluation_metrics.py

# Specific test
pytest tests/unit/test_evaluation_metrics.py::TestRetrievalMetrics::test_perfect_alignment
```

## Documentation

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve locally (auto-reload)
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

### Documentation Structure

- **Getting Started**: Installation, quickstart guides
- **Concepts**: Architecture and methodology explanations  
- **Advanced**: Deep-dive topics like determinism
- **API Reference**: Auto-generated from docstrings
- **Development**: Contributing and development guides

### Writing Documentation

- **Use clear, concise language**
- **Include code examples** that actually work
- **Add diagrams** for complex concepts (Mermaid supported)
- **Cross-reference** related sections
- **Test examples** in docstrings

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Clear description** of the problem
- **Minimal reproduction** case
- **Environment details** (Python version, OS, package versions)
- **Expected vs actual behavior**
- **Error messages** and stack traces

### Feature Requests

Use the feature request template and include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Proposed implementation** approach (if you have ideas)
- **Alternatives considered**

## Pull Request Guidelines

### Before Submitting

- [ ] **Fork the repository** and create a feature branch
- [ ] **Install pre-commit hooks**: `pre-commit install`
- [ ] **Write tests** for new functionality
- [ ] **Update documentation** for API changes
- [ ] **Run quality checks**: `make quality`
- [ ] **Check test coverage** remains >90%

### PR Description

Include in your PR description:

- **Summary** of changes made
- **Motivation** for the changes
- **Testing** approach and results
- **Breaking changes** (if any)
- **Related issues** (use "Fixes #123" syntax)

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Address feedback** promptly
4. **Squash commits** before merge (if requested)

## Architecture Guidelines

### Code Organization

```
src/histo_omics_lite/
├── __init__.py           # Public API exports
├── cli.py               # Command-line interface
├── data/                # Data loading and processing
├── models/              # Model architectures
├── training/            # Training loops and utilities
├── evaluation/          # Evaluation metrics and tools
├── inference/           # Inference and embedding generation
└── utils/               # Shared utilities (determinism, etc.)
```

### Design Principles

- **Modularity**: Clear separation of concerns
- **Testability**: Easy to unit test individual components
- **Configurability**: Use Hydra configs for flexibility
- **Reproducibility**: Deterministic by default
- **Performance**: Optimize for common use cases
- **Extensibility**: Easy to add new models/metrics

### API Design

- **Consistent naming**: Use clear, descriptive names
- **Type safety**: Full type annotations with mypy compliance
- **Error handling**: Informative error messages
- **Backwards compatibility**: Avoid breaking changes
- **Documentation**: Complete docstrings for public APIs

## Performance Considerations

### Optimization Guidelines

- **Profile before optimizing**: Use `cProfile` or `py-spy`
- **Vectorize operations**: Use PyTorch/NumPy operations
- **Memory efficiency**: Be mindful of large tensor allocations
- **CPU vs GPU**: Support both with automatic detection
- **Batch processing**: Default to reasonable batch sizes

### Benchmarking

- **Add benchmarks** for performance-critical code
- **Use `pytest-benchmark`** for microbenchmarks
- **Test on realistic data sizes**
- **Monitor memory usage**

## Security

### Security Guidelines

- **No hardcoded secrets**: Use environment variables
- **Input validation**: Validate all user inputs
- **Dependency security**: Keep dependencies updated
- **Code scanning**: We use Bandit for security analysis

### Reporting Security Issues

For security vulnerabilities, please email security@altalanta.ai instead of opening a public issue.

## Release Process

### For Maintainers

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run full CI**: `make ci`
4. **Create release tag**: `git tag v0.1.0`
5. **Push tag**: `git push origin v0.1.0`
6. **GitHub Actions** handles PyPI upload
7. **Create GitHub release** with changelog

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible new features
- **PATCH**: Backwards-compatible bug fixes

## Community

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). 

### Getting Help

- **GitHub Discussions**: For questions and community discussion
- **GitHub Issues**: For bug reports and feature requests  
- **Documentation**: Check docs first for common questions
- **Stack Overflow**: Tag questions with `histo-omics-lite`

### Recognition

All contributors will be recognized in our documentation and release notes. Significant contributors may be invited to become maintainers.

## Development Tips

### IDE Setup

For VS Code, use these extensions:
- Python
- Pylance  
- Ruff
- MyPy
- GitLens

### Debugging

```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or with ipdb for better interface
import ipdb; ipdb.set_trace()

# For PyTorch debugging
import torch
torch.autograd.set_detect_anomaly(True)
```

### Common Issues

- **Import errors**: Ensure you installed with `-e` flag
- **Type errors**: Run `mypy src/histo_omics_lite` to check
- **Test failures**: Run specific tests to isolate issues
- **Pre-commit fails**: Run `pre-commit run --all-files`

Thank you for contributing to Histo-Omics-Lite! 🎉