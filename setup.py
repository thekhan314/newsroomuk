import setuptools

REQUIRED_PACKAGES  = [
    'apache-beam',
    'google',
    'nltk',
    'pandas',
    'google-cloud-storage',
]


setuptools.setup(
    name='my_pipeline',
    version='0.0.0',
    packages=setuptools.find_packages(),
    install_requires = REQUIRED_PACKAGES
)
