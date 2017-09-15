########################################################
#                     GEVOLAB SETUP
########################################################
#                       WELCOME

# ADDRESS NEEDED INCLUDES AND LIBRARIES
import time
includes="-I/home/gf/work/forsat/geneva_works/gevolution/code/hdf5-1.8.18/hdf5/include"
libs="-L/home/gf/work/forsat/geneva_works/gevolution/code/hdf5-1.8.18/hdf5/lib"

print('####################   WELCOME   ####################')
print('#############   GEVOLAB IS SETTING UP   #############')
print
time.sleep(2)

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

# dependencies can be any iterable with strings, 
# e.g. file line-by-line iterator
dependencies = [
  'h5py>=2.6.0',
  'numpy>=1.11'
]

pkg_resources.require(dependencies)

import os
import sys
import glob
import site
import shutil
import subprocess

def get_data(path):
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_ROOT, 'static', path)

def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

try:
    from setuptools import setup
except ImportError:
    print('WARNING: could not import setuptools!')
    print('         make sure h5py & numpy is installed!')
    from distutils.core import setup

package_name = 'gevolab'
package_version = '0.1'

setup(name=package_name,
      version=package_version,
      description='The funniest joke in the world',
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=[package_name],
#    include_package_data=True,
#      data_files=data,
      zip_safe=False)

subprocess.call(['rm','-r','build','dist','gevolab.egg-info'])

try:
    from distutils.sysconfig import get_python_lib
    pack_add = get_python_lib()
    pack_add = glob.glob(pack_add+'/'+package_name+'-'+package_version+'*')[0]+'/'
except:
    pack_add = site.getsitepackages()[0]
    pack_add = glob.glob(pack_add+'/'+package_name+'-'+package_version+'*')[0]+'/'

data_dir = pack_add+'static'

src = get_data('')
copyDirectory(src, data_dir)

exe_dir = pack_add+'exe'

if not os.path.exists(exe_dir):
    os.makedirs(exe_dir)

code="pre-ext.cpp"

if libs!='':
    lib_add = pack_add+'gevolab/lib_init'
    f = open(lib_add,'w') 
    f.write(libs.replace('-L','')) 
    f.close()

LATFEILD = "-I"+data_dir+"/LATfield2"

code = data_dir+"/pre-ext.cpp"
subprocess.call(['mpic++','-o',exe_dir+'/pre-ext',code,'-DHDF5','-std=c++11','-lm','-lhdf5',LATFEILD,includes,libs])

code = data_dir+"/pre-h5_gen.cpp"
subprocess.call(['mpic++','-o',exe_dir+'/pre-h5_gen',code,'-DHDF5','-std=c++11','-lm','-lhdf5',LATFEILD,includes,libs])

gevolution = 'gevolution/'
gevolution = data_dir+"/"+gevolution

if not os.path.exists(gevolution+'makefile.in'):
    shutil.copyfile(gevolution+'makefile', gevolution+'makefile.in')

exc = '# add the path to LATfield2 and other libraries (if necessary)'
inc = LATFEILD+' '+includes+' '+libs

f = open(gevolution+'makefile.in','r') 
makefile = f.read() 

f.close()
f = open(gevolution+'makefile','w') 
f.write(makefile.replace(exc,inc)) 
f.close()

subprocess.call(['make','-C',gevolution])


rockstar = 'rockstar/'
rockstar = data_dir+"/"+rockstar
subprocess.call(['make','-C',rockstar])

# Installing pygadgetreader

os.chdir('./gevolab/rthompson')

setup(name='pyGadgetReader',
      version='2.6',
      description='module to read all sorts of gadget files',
      author='Robert Thompson',
      author_email='rthompsonj@gmail.com',
      url='https://bitbucket.org/rthompson/pygadgetreader',
      packages=['readgadget','readgadget.modules','pygadgetreader'],
      #scripts=['bin/test.py'],
      install_requires=['h5py>=2.2.1','numpy>=1.7'],
      zip_safe=False)

subprocess.call(['rm','-r','build','dist','pyGadgetReader.egg-info'])
