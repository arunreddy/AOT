# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
  name='aot',
  version='1.0.0',
  description='Adaptive Off-the-shelf classifier',
  long_description='LONG Description',
  author='Arun Reddy Nelakurthi',
  author_email='arunreddy.nelakurthi@gmail.com',
  url='https://github.com/arunreddy/aot',
  license='MIT License',
  packages=find_packages(exclude=('tests', 'docs')),
  entry_points={
    'console_scripts': ['aot=aot.cli:main'],
  }
)
