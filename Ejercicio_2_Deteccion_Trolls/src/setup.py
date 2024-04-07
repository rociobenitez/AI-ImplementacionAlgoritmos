import setuptools

REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.24.0',
    'tensorflow==2.8.0',
    'gensim==3.6.0',
    'fsspec==0.8.4',
    'gcsfs==0.7.1',
    'numpy==1.20.0',
    'nltk==3.5',
    'textblob==0.15.3',
]

setuptools.setup(
    name='twitchstreaming',
    version='0.0.1',
    description='Troll detection with Apache Beam',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)