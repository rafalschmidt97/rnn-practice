from setuptools import setup, find_packages

setup(
    name='rnn-practice',
    packages=find_packages(),
    install_requires=[
        'tensorflow', 'h5py', 'graphviz', 'pydot', 'keras', 'numpy', 'pandas', 'scikit-learn', 'matplotlib'
    ],
)
