#!/usr/bin/env python3

import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

__author__ = 'Chace Ashcraft Kiran Karra'
__email__ = 'chace.ashcraft@jhuapl.edu, kiran.karra@jhuapl.edu'
__version__ = '0.1.1'

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    install_requires = ['numpy==1.17.4',
                        'tqdm>=4.32.1',
                        'torch>=1.4.0',
                        'torch-ac==1.1.0',
                        'gym==0.15.3',
                        'atari_py~=0.2.0',
                        'opencv-python',
                        ]

setuptools.setup(
    name='trojai_rl',
    version=__version__,

    description='TrojAI RL model generation library',
    long_description=long_description,
    long_description_content_type="text/markdown",

    url='https://github.com/trojai/trojai_rl',

    author=__author__,
    author_email=__email__,

    license='Apache License 2.0',

    python_requires='>=3.7',
    packages=['trojai_rl',
              'trojai_rl.datagen',
              'trojai_rl.datagen.envs',
              'trojai_rl.modelgen',
              'trojai_rl.modelgen.architectures'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='reinforcement-learning trojan adversarial',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=install_requires,

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': ['nose'],
        'stable_baselines': [
            'stable_baselines==2.8.0',
            'tensorflow==1.14; sys_platform=="darwin"',  # no TF GPU support on MAC :/
            'tensorflow-gpu==1.14; platform_system=="Linux"',
            'tensorflow-gpu==1.14; platform_system=="Windows"'
        ]
    },

    scripts=["scripts/wrapped_boxing.py"],

    zip_safe=False
)
