from setuptools import setup, find_packages

setup(
    name='matsindy',
    version='1.0',
    description='Sparse Identification of Nonlinear Dynamics for matrix variables',
    author='Etienne Rognin',
    author_email='ecr43@cam.ac.uk',
    packages=find_packages(include=['matsindy', 'matsindy.*'])
)
