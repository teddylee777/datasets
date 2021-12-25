from setuptools import setup, find_packages

setup(
    name='teddynote',
    version='0.1.6',
    description='datasets and tutorial package made and maintained by TeddyNote',
    author='teddylee777',
    author_email='teddylee777@gmail.com',
    url='https://github.com/teddylee777/datasets',
    install_requires=['tqdm', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'optuna'],
    packages=find_packages(exclude=[]),
    keywords=['teddynote', 'teddylee777', 'python datasets', 'python tutorial', 'machine learning', 'deep learning', 'optuna'],
    python_requires='>=3',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
