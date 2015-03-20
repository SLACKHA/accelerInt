#! /usr/bin/env python2.7
import math
import subprocess
import os, glob
import sys

#force remake
subprocess.call(['make', '-j24', 'DEBUG=FALSE', 'FAST_MATH=FALSE', 'IGN=TRUE', 'PRINT=FALSE', 'LOG_OUTPUT=TRUE'])

all_exes = []
for file in glob.glob('*-int*'):
	all_exes.append(file)


for exe in all_exes:
	subprocess.call([os.path.join(os.getcwd(), exe)])
