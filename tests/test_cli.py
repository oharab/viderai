"""Tests for CLI functionality."""

import logging
import pytest
from click.testing import CliRunner

from viderai.cli import setup_logging


def test_setup_logging_verbose():
    """Test verbose logging configuration."""
    setup_logging(verbose=True)
    
    logger = logging.getLogger('viderai')
    assert logger.level == logging.DEBUG


def test_setup_logging_quiet():
    """Test quiet logging configuration."""
    setup_logging(verbose=False)
    
    logger = logging.getLogger('viderai')
    assert logger.level == logging.WARNING
    
    # Test ultralytics logger is suppressed
    ultralytics_logger = logging.getLogger('ultralytics')
    assert ultralytics_logger.level == logging.ERROR


def test_cli_help():
    """Test CLI help command."""
    from viderai.cli import main
    
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    
    assert result.exit_code == 0
    assert 'VIDEO_PATH' in result.output
    assert '--verbose' in result.output
    assert '--quiet' in result.output