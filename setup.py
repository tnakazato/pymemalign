#from distutils.core import setup
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