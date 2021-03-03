from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="stratified_group_kfold",
      version="0.1",
      install_requires=requirements,
      packages=find_packages())
