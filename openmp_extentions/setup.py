#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

roi_avg_module = Extension('_roi_avg',
                           sources=['roi_avg_wrap.cxx', 'roi_avg.cpp'],
                           include_dirs=[numpy.get_include(), '/.'],
                           extra_compile_args=["-fopenmp"],
                           extra_link_args=['-lgomp'],
                           swig_opts=['-threads']
                           )

setup(name='roi_avg',
      version='0.1',
      author="Matt Howa",
      description="""Average ROIs for cell profiling""",
      ext_modules=[roi_avg_module],
      py_modules=["roi_avg"],
      )
