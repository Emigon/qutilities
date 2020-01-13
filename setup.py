from setuptools import setup, find_packages

setup(name='qutilities',
      version='0.0.1',
      description='Utilities for measuring Q-factors',
      url='https://github.com/Emigon/qutilities',
      author='Daniel Parker',
      author_email='danielparker@live.com.au',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.16.3',
          'sympy>=1.5.1',
          'pandas>=0.24.0'
          'fitkit>=0.1.0',
          'matplotlib',
        ])
