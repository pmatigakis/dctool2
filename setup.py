from setuptools import setup, find_packages

setup(
    name='dctool2',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "beautifulsoup4==4.5.1",
        "luigi==2.3.3",
        "snakebite==2.11.0",
        "numpy==1.11.1",
        "scipy==0.18.0",
        "scikit-learn==0.17.1"
    ]
)
