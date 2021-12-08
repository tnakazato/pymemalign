# Copyright (C) 2021
# National Astronomical Observatory of Japan
# 2-21-1, Osawa, Mitaka, Tokyo, 181-8588, Japan.
#
# This file is part of pymemalign.
#
# pymemalign is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# pymemalign is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with pymemalign.  If not, see <https://www.gnu.org/licenses/>.
import setuptools
from distutils.extension import Extension

def get_numpy_include_dir():
    import numpy
    return numpy.get_include()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(where='src')
print(packages)

setuptools.setup(
    name='pymemalign',
    version='0.1.0',
    author='Takeshi Nakazato',
    author_email='takeshi.nakazato@nao.ac.jp',
    description='Python wrapper of posix_memalign',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tnakazato/pymemalign',
    classifiers=[
    ],
    install_requires=['numpy'],
    extras_require={
        'test': ["pytest", "memory_profiler"],
    },
    package_dir={'pymemalign': 'src'},
    packages=['pymemalign'],
    python_requires='>=3.6',
    ext_modules=[
        Extension(
            'pymemalign._pymemalign', ['src/pymemalign.cc'],
            include_dirs=[get_numpy_include_dir()],
            extra_compile_args=['-std=c++11']
        )
    ],
)