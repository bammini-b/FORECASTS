from setuptools import setup, find_packages

setup(
    name='forecasts',
    version='0.1.0', # Initial version. Increment this for future releases.
    author='Thanabammini Balasaravanan', # Updated name
    author_email='bamminib@hotmail.com', # Updated email
    description='A Python package for classifying cellular states in transcriptomics data.',
    long_description=open('README.md').read(), # Assumes you will create a README.md later
    long_description_content_type='text/markdown',
    url='https://github.com/bammini-b/FORECASTS', 
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.20.0',
        'scanpy>=1.9.0',
        'qnorm>=0.2.0',
        'scikit-learn>=1.0.0',
        'anndata>=0.9.0',
        'squidpy', # Keeping for now as requested
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Placeholder, see license recommendation below
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8', # Specify minimum Python version required
)