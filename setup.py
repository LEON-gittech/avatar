from setuptools import setup, find_packages

setup(
    name="avatar",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "tenacity",
        "numpy",
        "torch",
        "stark-qa"
    ]
) 