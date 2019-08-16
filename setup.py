from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.readlines()

setup(
    name='dctool2',
    version='0.1.0',
    author="Panagiotis Matigakis",
    author_email="pmatigakis@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires
)
