from setuptools import setup, find_packages

setup(
    name='rnn-practice',
    packages=find_packages(),
    install_requires=[
        'tensorflow<2.1.0', 'h5py', 'graphviz', 'pydot', 'keras', 'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'python-aqi'
    ],
)
