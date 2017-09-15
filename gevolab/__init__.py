from . import gev

import os
import sys

libs_add = __file__.split('/')[:-2]
libs_add = '/'.join(libs_add)+'/gevolab/lib_init'
f = open(libs_add,'r') 
libs = f.read() 

try:
	os.environ['LD_LIBRARY_PATH'] += ":"+libs
except:
	os.environ['LD_LIBRARY_PATH'] = ":"+libs

f.close()

#libs = sys.executable.split('/')[:-2]
#libs = '/'.join(libs)+'/lib'
#os.environ['LD_LIBRARY_PATH'] += ":"+libs

