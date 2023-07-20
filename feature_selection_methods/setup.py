from setuptools import setup, find_packages

setup(
    name='feature_selection_methods',
    version='1.0.0',
    description='A package for various feature selection methods',
    author='Siddharth Verma',
    author_email='Siddharth1.Verma@zmail.ril.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'scipy'
    ]
)
