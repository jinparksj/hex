# setup.py
from setuptools import setup, find_packages

setup(
    name='schema',
    packages=[package for package in find_packages()
              if package.startswith('schema')],
    version='0.0.1',)
