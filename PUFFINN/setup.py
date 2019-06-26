import os
import sys

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', format='md')
except (IOError, ImportError):
    long_description = 'test' #open('README.md').read()

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

extra_args = ['-std=c++14', '-march=native', '-O3', '-fopenmp']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    '_puffinn',
    sources=['python/wrapper/python_wrapper.cpp'],
    extra_compile_args=extra_args,
    extra_link_args=['-fopenmp'],
    include_dirs=['include', 'external/pybind11/include', 'libs'])

setup(
    name='PUFFINN',
    version='0.1',
    author='Michael Erik Vesterli, Martin Aumüller',
    author_email='maau@itu.dk',
    url='https://github.com/',
    description=
    'High-Dimenional Similarity search with guarantees based on Locality-Sensitive Hashing (LSH)',
    #long_description=long_description,
    license='MIT',
    keywords=
    'nearest neighbor search similarity lsh locality-sensitive hashing cosine distance euclidean',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[module])
