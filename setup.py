from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='mtbi_meeg',
    version='0.0.1',
    description='Pipeline to analyze EEG and MEG data and determine minor Trauma Brain Injuries',
    author='Verna Heikkinen, Estanislao Porta, Aino Kuusi',
    author_email='estanislao.porta@aalto.fi',
    install_requires=requirements,
    url='https://github.com/BioMag/mtbi_meeg',
    package_dir = {'': 'src'},
    packages = ['analysis', 'processing'],
    classifiers=[
	'Programming Language :: Python :: 3',
	'License :: OSI Approved :: MIT License',
	'Operating System :: OS Independent',
	],
)
