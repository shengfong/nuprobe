from setuptools import setup
from os import path

from nuprobe.version import __version__ as version

with open("requirements.txt", "r") as f:
    requirements = f.read().split()

setup(
    name='nuprobe',
    version=version,
    python_requires='>=3.8',
    description="Neutrino oscillation as a Probe of New Physics",
    url="https://github.com/shengfong/nuprobe",
    author="Chee Sheng Fong",
    license='GPL3',
    packages=[
        'nuprobe'
    ],
    install_requires=requirements)
