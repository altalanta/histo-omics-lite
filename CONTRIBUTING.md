# Contributing to histo-omics-lite

We welcome contributions to histo-omics-lite! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/altalanta/histo-omics-lite
   cd histo-omics-lite
   ```

2. **Install development dependencies**:
   ```bash
   make setup  # or pip install -e .[dev]
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Quality Standards

All contributions must pass our quality gates:

- **Formatting**: Code must be formatted with `ruff format`
- **Linting**: Code must pass `ruff check` with our configuration
- **Type Checking**: Code must pass `mypy` strict mode on `src/`
- **Testing**: New features must include tests with ≥90% coverage
- **Documentation**: Public APIs must be documented

### Running Quality Checks

```bash
# Format code
make format

# Check linting
make lint

# Run type checking
make type

# Run tests with coverage
make test

# Run smoke tests (fast end-to-end validation)
make smoke
```

### Testing

We maintain different levels of testing:

1. **Unit Tests**: Fast, isolated tests of individual components
2. **Integration Tests**: Tests of component interactions
3. **Smoke Tests**: Fast end-to-end pipeline validation (≤5 minutes)
4. **Golden Tests**: Deterministic output validation with hash checking

#### Adding Tests

- Place unit tests in `tests/` with descriptive names
- Use pytest fixtures for common setup
- Mock external dependencies appropriately
- Include both positive and negative test cases
- Test edge cases and error conditions

#### Test Naming Convention

```
test_[component]_[scenario]_[expected_outcome].py
```

Examples:
- `test_cli_data_make_creates_synthetic_dataset.py`
- `test_determinism_same_seed_produces_identical_results.py`
- `test_metrics_bootstrap_confidence_intervals_valid_bounds.py`

### Documentation

Documentation is built with MkDocs Material and includes:

- **Quickstart Guide**: Getting started instructions
- **Concepts**: Architecture and methodology explanation
- **API Reference**: Auto-generated from docstrings
- **Determinism Guide**: Reproducibility best practices

#### Writing Documentation

- Use clear, concise language
- Include runnable code examples
- Add docstrings to all public functions/classes
- Follow Google-style docstring format
- Test code examples in CI to prevent rot

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**:
   ```bash
   make lint type test smoke
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Guidelines

We follow conventional commits:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or modifying tests
- `chore:` - Maintenance tasks

Examples:
- `feat: add bootstrap confidence intervals for metrics`
- `fix: resolve determinism issue in UMAP embeddings`
- `docs: add quickstart guide for CLI usage`

### Configuration Changes

When modifying Hydra configurations:

1. **Validate configs** with pydantic schemas
2. **Test all profiles** (fast_debug, cpu_small, gpu_quick)
3. **Update documentation** for new parameters
4. **Maintain backward compatibility** when possible

### Adding New CLI Commands

When adding CLI commands:

1. **Use Typer** for consistent interface
2. **Add JSON output mode** with `--json` flag
3. **Include help text** and examples
4. **Support common flags**: `--seed`, `--cpu/--gpu`, `--deterministic`
5. **Validate inputs** early with clear error messages

### Performance Guidelines

- **CPU Performance**: Smoke tests must complete in ≤5 minutes
- **Memory Usage**: Keep datasets ≤100MB for CI
- **Dependencies**: Minimize runtime dependencies
- **Determinism**: Ensure reproducible results across runs

### Release Process

Releases are automated through GitHub Actions:

1. **Version bump**: Update version in `pyproject.toml` and `__init__.py`
2. **Update CHANGELOG.md**: Document changes following Keep a Changelog
3. **Tag release**: `git tag v0.x.y && git push origin v0.x.y`
4. **CI/CD**: Automatically builds and publishes to PyPI

### Getting Help

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Code Review**: Maintainers will review PRs and provide feedback

### Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming environment for contributors of all experience levels.

## Recognition

Contributors will be acknowledged in:
- Release notes
- GitHub contributors page
- Documentation credits

Thank you for contributing to histo-omics-lite!