from setuptools import find_packages
from setuptools import setup

setup(
    name='rl_finance',
    version='0.1.0',
    author='Trung Dang',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='trungdangminh14012004@gmail.com',
    description='Reinforcement Learning for Quantitative Trading',
    install_requires=['ml_logger==0.8.117',
                      'ml_dash==0.3.20',
                      'jaynes>=0.9.2',
                      'params-proto==2.10.5',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.26.4',
                      ]
)