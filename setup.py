#!/usr/bin/env python

import os

from setuptools import find_packages, setup


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


long_description = get_long_description()

with open("requirements.txt", "r") as ff:
    requirements = ff.readlines()
setup(
    name="gwpopulation",
    description="Unified population inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ColmTalbot/gwpopulation",
    author="Colm Talbot",
    author_email="talbotcolm@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["test", "venv", "priors"]),
    package_dir={"gwpopulation": "gwpopulation"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
