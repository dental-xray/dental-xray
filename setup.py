from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="disease-recognition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements
)
