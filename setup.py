"""
Author: Travis Hammond
©️ 2022 The Johns Hopkins University Applied Physics Laboratory LLC.
"""

import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='hapi_nn',
    version='0.1.0',
    author='Travis Hammond',
    author_email='Travis.Hammond@jhuapl.edu',
    description='HAPI-NN allows interfacing of HAPI with TensorFlow and '
                'PyTorch to rapidly create deep neural network models for '
                'predicting and forecasting.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hapi-server/application-neuralnetwork-python',
    license='LICENSE',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=['numpy>=1.22', 'matplotlib>=3.4',
                      'hapiclient>=0.2'],
)
