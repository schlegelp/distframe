from setuptools import setup, find_packages
import re


VERSIONFILE = "distframe/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name='distframe',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='DataFrame-like wrapper around condensed distance matrices.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/schlegelp/distframe',
    project_urls={
     "Documentation": "https://github.com/schlegelp/distframe",
     "Source": "https://github.com/schlegelp/distframe",
     "Changelog": "https://github.com/schlegelp/distframe",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='DataFrame pandas distances matrix condensed',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=requirements,
    extras_require={},
    python_requires='>=3.7',
    zip_safe=False,
    include_package_data=True
)
