#!/usr/bin/env python

"""The setup script."""

import io
from os import path as op
from setuptools import setup, find_packages

'''
# avoid crash because of Chinese like this: https://pastebin.com/drsw2DEL
with open('README.md') as readme_file:
    readme = readme_file.read()
''' 

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Wenyu Ouyang",
    author_email='hust2014owen@gmail.com',
    python_requires='>=3.9',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="A python project to deal with hydrological datasources",
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="BSD license",
    # long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='hydrodatasource',
    name='hydrodatasource',
    packages=find_packages(include=['hydrodatasource', 'hydrodatasource.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/iHeadWater/hydrodatasource',
    version='0.0.5',
    zip_safe=False,
)
