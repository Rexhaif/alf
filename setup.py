import re

import setuptools


with open("alf/__init__.py", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)
    if version is None:
        raise ImportError("Could not find __version__ in alf/__init__.py")

setuptools.setup(
    name="alf",
    version=version,
    packages=setuptools.find_packages(include=["alf*"]),
    url="http://github.com/Rexhaif/alf/",
    author="Daniil Larionov",
    author_email="rexhaif.io@gmail.com",
    description="Active Learning-based hyperparameter optimization",
    long_description="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    platforms=["Linux", "OS-X", "Windows"],
    license="BSD",
    keywords="active learning optimization hyperparameter model selection",
    include_package_data=True,
    install_requires=[
        "scikit-learn==0.24.1",
        "numpy==1.20.2",
        "pandas==1.2.3",
        "modAL==0.4.1",
        "rich==10.0.1"
    ],
    tests_require=["pytest"]
)