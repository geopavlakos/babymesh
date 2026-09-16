from setuptools import find_packages, setup

setup(
    name="slahmr-mview",
    packages=find_packages(
        where="slahmr",
    ),
    package_dir={"": "slahmr"},
)
