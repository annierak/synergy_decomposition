from setuptools import setup
from setuptools import find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='synergy_decomposer',
    version='0.0.0',
    description='Muscle synergy extraction implementations',
    long_description=__doc__,
    author='Annie Rak & Alysha de Souza',
    author_email='arak@caltech.edu',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Biology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    packages=find_packages(exclude=['examples',]),
)
