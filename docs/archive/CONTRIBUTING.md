# Contributing to Magnus Trading Platform

Thank you for your interest in contributing to Magnus! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [NO_DUMMY_DATA_POLICY](#no_dummy_data_policy)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Our Standards

**Positive behaviors include:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**

- Harassment, discrimination, or derogatory comments
- Trolling, insulting comments, or personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.9 or higher installed
- PostgreSQL 14 or higher
- Redis server
- Git for version control
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/your-username/WheelStrategy.git
cd WheelStrategy
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/original-owner/WheelStrategy.git
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Documentation**: Improve or expand documentation
- **Testing**: Write or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code quality without changing functionality

### Contribution Workflow

1. **Check existing issues**: Look for existing issues or create a new one to discuss your idea
2. **Create a branch**: Create a feature branch from `main`
3. **Make changes**: Implement your changes following our coding standards
4. **Write tests**: Add tests for new functionality
5. **Update documentation**: Update relevant documentation
6. **Submit PR**: Submit a pull request for review

## Development Setup

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Set Up Database

```bash
# Create test database
createdb magnus_test

# Run migrations
python scripts/setup_database.py
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env.test

# Edit with test credentials
# Use separate credentials from production!
```

### 5. Run Tests

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_premium_scanner.py
```

## Coding Standards

### NO_DUMMY_DATA_POLICY

**CRITICAL**: This project follows a strict NO DUMMY DATA policy. Review [NO_DUMMY_DATA_POLICY.md](./NO_DUMMY_DATA_POLICY.md) before contributing.

**Key Rules:**

1. **NEVER** use hardcoded fake data (fake balances, test trades, sample positions)
2. **ALWAYS** use real API data or show empty states
3. **ALWAYS** use `0`, `None`, `[]`, or `{}` as defaults - not fake numbers
4. **NEVER** commit test/dummy data to the database
5. **ALWAYS** validate data comes from legitimate sources (APIs, user input)

**Example - BAD (DO NOT DO THIS):**

```python
# ❌ NEVER use fake default values
balance = account_data.get('balance', 100000)  # Fake $100k
positions = [
    {'symbol': 'NVDA', 'premium': 610}  # Test data
]
```

**Example - GOOD (DO THIS):**

```python
# ✅ Check for real data, show empty state if missing
if account_data and 'balance' in account_data:
    balance = float(account_data['balance'])
    display_metrics(balance)
else:
    st.info("Connect to Robinhood to see your balance")

# ✅ Empty list if no real data
positions = fetch_positions_from_api() or []
```

### Python Style Guide

We follow PEP 8 with some modifications:

**Code Formatting:**

- Use `black` for automatic formatting (line length: 100)
- Use `flake8` for linting
- Use `mypy` for type checking
- Use type hints for all function parameters and return values

**Naming Conventions:**

- Classes: `PascalCase` (e.g., `PremiumScanner`)
- Functions/Methods: `snake_case` (e.g., `calculate_premium`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_STRIKE_PRICE`)
- Private methods: `_leading_underscore` (e.g., `_internal_method`)

**Example:**

```python
from typing import List, Dict, Optional
from datetime import datetime


class OptionAnalyzer:
    """Analyzes option opportunities for wheel strategy."""

    MAX_PRICE: float = 50.0
    MIN_VOLUME: int = 100

    def __init__(self, max_price: Optional[float] = None) -> None:
        """Initialize analyzer with optional max price."""
        self.max_price = max_price or self.MAX_PRICE

    def analyze_opportunity(
        self,
        symbol: str,
        strike: float,
        expiration: datetime
    ) -> Dict[str, any]:
        """Analyze a specific option opportunity.

        Args:
            symbol: Stock ticker symbol
            strike: Option strike price
            expiration: Option expiration date

        Returns:
            Dictionary containing analysis results
        """
        # Implementation
        return {}

    def _calculate_score(self, data: Dict[str, any]) -> float:
        """Internal method to calculate opportunity score."""
        # Implementation
        return 0.0
```

### Documentation Standards

**Docstrings:**

- Use Google-style docstrings for all functions and classes
- Document all parameters, return values, and exceptions
- Include usage examples for complex functions

**Comments:**

- Write clear, concise comments explaining "why" not "what"
- Update comments when code changes
- Remove commented-out code before committing

**README Files:**

- Each feature should have a `README.md` in `features/[feature_name]/`
- Include: Overview, How to Use, Screenshots, Troubleshooting
- Keep documentation up-to-date with code changes

## Testing Requirements

### Test Coverage

- All new features must include tests
- Maintain minimum 80% code coverage
- Test both success and failure cases
- Test edge cases and boundary conditions

### Test Structure

```python
import pytest
from src.premium_scanner import PremiumScanner


class TestPremiumScanner:
    """Tests for PremiumScanner class."""

    @pytest.fixture
    def scanner(self):
        """Create scanner instance for testing."""
        return PremiumScanner(max_price=50.0)

    def test_scan_returns_valid_opportunities(self, scanner):
        """Test that scan returns properly formatted opportunities."""
        symbols = ['AAPL', 'MSFT']
        results = scanner.scan_premiums(symbols, dte=30)

        assert isinstance(results, list)
        assert all('symbol' in r for r in results)
        assert all('premium_pct' in r for r in results)

    def test_scan_filters_by_max_price(self, scanner):
        """Test that scan respects max price filter."""
        symbols = ['AAPL', 'AMZN']  # AMZN likely > $50
        results = scanner.scan_premiums(symbols, dte=30)

        # All results should be under max_price
        assert all(r['stock_price'] <= 50.0 for r in results)

    def test_scan_handles_invalid_symbol(self, scanner):
        """Test that scan handles invalid symbols gracefully."""
        symbols = ['INVALID123']
        results = scanner.scan_premiums(symbols, dte=30)

        # Should return empty list, not raise exception
        assert results == []
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_premium_scanner.py

# Run specific test
pytest tests/test_premium_scanner.py::TestPremiumScanner::test_scan_returns_valid_opportunities

# Run tests matching pattern
pytest -k "premium"
```

### Test Database

- Use separate test database (`magnus_test`)
- Clean up test data after each test
- Use fixtures for common setup
- Never use production database for tests

## Pull Request Process

### Before Submitting

1. **Update documentation**: Ensure README and feature docs are current
2. **Run tests**: All tests must pass (`pytest`)
3. **Check coverage**: Maintain or improve code coverage
4. **Format code**: Run `black`, `flake8`, and `mypy`
5. **Update CHANGELOG**: Add entry describing your changes
6. **Verify NO_DUMMY_DATA_POLICY**: Ensure no fake/test data

### PR Checklist

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Compliance
- [ ] Code follows NO_DUMMY_DATA_POLICY
- [ ] Code formatted with black
- [ ] Linting passes (flake8)
- [ ] Type checking passes (mypy)
- [ ] Documentation updated
- [ ] CHANGELOG updated

## Screenshots (if applicable)
Add screenshots of UI changes
```

### PR Guidelines

- **Title**: Use clear, descriptive title (e.g., "Add multi-expiration support to premium scanner")
- **Description**: Explain what changes you made and why
- **Single Purpose**: Each PR should address one issue/feature
- **Small PRs**: Keep PRs focused and reasonably sized
- **Tests**: Include tests demonstrating the fix/feature works
- **Documentation**: Update relevant documentation

### Review Process

1. Automated checks must pass (tests, linting, coverage)
2. At least one maintainer approval required
3. Address review feedback promptly
4. Squash commits before merge (if requested)
5. Maintainer will merge approved PRs

## Reporting Bugs

### Before Reporting

1. Check existing issues to avoid duplicates
2. Try to reproduce the bug in latest version
3. Gather relevant information (OS, Python version, logs)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Enter '...'
4. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python Version: [e.g., 3.11.5]
- Magnus Version: [e.g., 1.2.0]

**Additional context**
Any other relevant information.

**Error logs**
```
Paste relevant error logs here
```
```

## Suggesting Features

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Problem it Solves**
Explain the problem this feature would solve.

**Proposed Solution**
Describe how you envision the feature working.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other relevant information, mockups, or examples.

**Willing to Contribute**
- [ ] I'm willing to implement this feature
- [ ] I can help with testing
- [ ] I can help with documentation
```

### Feature Discussion

- Features are discussed in GitHub Issues first
- Large features may require a design document
- Maintainers will provide feedback and direction
- Implementation begins after consensus is reached

## Development Best Practices

### Code Quality

- Write self-documenting code with clear variable names
- Keep functions small and focused (single responsibility)
- Avoid deep nesting (max 3-4 levels)
- Use type hints consistently
- Handle errors gracefully with try/except
- Log important operations and errors

### Security

- Never commit credentials or API keys
- Use environment variables for configuration
- Validate all user input
- Sanitize data before database insertion
- Use parameterized queries (prevent SQL injection)
- Review third-party dependencies for vulnerabilities

### Performance

- Use database indexes appropriately
- Cache expensive operations (Redis)
- Avoid N+1 query problems
- Use async/await for I/O operations
- Profile code before optimizing
- Document any performance trade-offs

### Database Changes

- Create Alembic migrations for schema changes
- Test migrations up and down
- Include rollback strategy
- Document schema changes in DATABASE_SCHEMA.md
- Never drop tables with data without backup

## Questions?

If you have questions about contributing:

1. Check the documentation in `features/` directory
2. Review existing issues and PRs
3. Ask in GitHub Discussions
4. Create an issue with the "question" label

## Recognition

Contributors are recognized in:

- CHANGELOG.md for their contributions
- README.md acknowledgments section
- GitHub contributors page

Thank you for contributing to Magnus Trading Platform!
