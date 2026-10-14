from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas>=2.3,<3',
        'scipy',
        'pyarrow',
        'scikit-learn',
        'torch',
        'anndata>=0.12,<0.13',
        'h5py',
        'wandb',
        'requests',
        'ipykernel',
    ],
    entry_points={
        'console_scripts': [
            'mldev-train=main:main',  # Assumes main.py has a main()
        ],
    },
    author='Jonathan Ngai',
    description='Modular ML development template for deep learning projects',
    url='https://github.com/J-Ngaiii/mldev-template',
    python_requires='>=3.9,<3.12',
)