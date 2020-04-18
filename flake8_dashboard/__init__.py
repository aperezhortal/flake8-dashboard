# -*- coding: utf-8 -*-
"""
flake8-dashboard
================

A flake8 plugin to generate a HTML dashboard with all the flake8 violations.
"""

__version__ = "0.1.3"

from .plugin import DashboardReporter

__all__ = ("DashboardReporter",)
