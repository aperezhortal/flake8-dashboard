# -*- coding: utf-8 -*-
"""Setuptools script for flake8-dashboard."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'pygments>=2.2.0',
    'flake8>=3.3.0',
    'plotly',
    'beautifulsoup4',
    'jsmin',
    'jinja2',
    'requests',
    'pandas',
    'astroid>=2.2.5'
]

setup(
    name='flake8-dashboard',
    version='0.1.0',
    description="Generate different reports of flake8 violations",
    long_description=readme,
    author="Andres Perez Hortal",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="Apache Software License 2.0",
    entry_points={
        'flake8.report': [
            'dashboard = flake8_dashboard:DashboardReporter',
        ]
    },
    python_requires='>=3.6',
    zip_safe=False,
    keywords='flake8 dashboard html',
    url="https://aperezhortal.github.io/flake8-dashboard/",
    classifiers=[
        'Framework :: Flake8',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
