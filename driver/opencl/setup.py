from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

path = os.path.dirname(__file__)
topdir = os.path.abspath(os.path.join(path, os.pardir, os.pardir))
libdir = os.path.join(topdir, 'lib')
includes = [os.path.join(topdir, 'generic', 'opencl'),
            os.path.join(topdir, 'interface', 'opencl'),
            os.path.join(topdir, 'rkf45', 'opencl'),
            os.path.join(topdir, 'paths')]

ext_module = Extension('pyccelerInt_ocl',
                       sources=[os.path.join(path, 'pyccelerInt_ocl.pyx')],
                       include_dirs=includes + [numpy.get_include()],
                       language="c++",
                       extra_compile_args=['-frounding-math', '-fsignaling-nans',
                                           '-std=c++11', '-fopenmp'],
                       libraries=['accelerint_problem_opencl', 'accelerint_opencl'],
                       runtime_library_dirs=[libdir],
                       library_dirs=[libdir])

setup(
    name='pyccelerInt_ocl',
    ext_modules=cythonize(ext_module)
)
