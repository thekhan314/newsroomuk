import setuptools

REQUIRED_PACKAGES  = [
    'apache-beam',
    'apache-beam[gcp]',
    'google',
    'nltk',
    'pandas',
    'google-cloud-storage',
    'numpy<1.20.0'
]


setuptools.setup(
    name='uk_pipeline',
    version='0.0.0',
    setup_requires = REQUIRED_PACKAGES,
    install_requires = REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True
    
)
