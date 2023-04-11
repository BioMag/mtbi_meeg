from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mtbi_meeg',
    version='0.0.1',
    description='Pipeline to analyze EEG and MEG data and determine minor Trauma Brain Injuries',
    author=['Verna Heikkinen', 'Estanislao Porta', 'Aino Kuusi'],
    author_email=['verna.heikkinen@example.com', 'estanislao.porta@aalto.fi'],
    install_requires=requirements,
    url='githubURL',
    package_dir = {'': 'src'},
    packages = ['analysis', 'processing'],
    classifiers=[
	'Programming Language :: Python :: 3',
	'License :: OSI Approved :: MIT License',
	'Operating System :: OS Independent',
	],
)
