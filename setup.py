import os

from setuptools import find_packages, setup


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f if line.strip() != '']


setup(
    name='JaxSeq',
    version='1.0.0',
    description='Implementations of LLM training in JAX.',
    url='https://github.com/Sea-Snell/JaxSeq2',
    author='Charlie Snell',
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt'),
    license='LICENCE',
)
