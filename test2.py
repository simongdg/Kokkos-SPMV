from subprocess import call
from sys import argv
from itertools import izip
import os

path = "./"
files = []
script, filename, paramfile = argv
#txt = open(filename)

mm_path = "/home/crtrott/matrices/MM/"

print "Generating results for all matrices in  %r:" % filename
print "Reading parameter file %r:" % paramfile

for fileline, paramline in izip(open(filename), open(paramfile)):
 	call("srun ./test_matvec.cuda-new" + paramline.rstrip() + " -fb " + mm_path + fileline, shell=True),
	for i in os.listdir(path):
    		if os.path.isfile(os.path.join(path,i)) and 'k40' in i:
			call("mv " + i + " " + fileline.replace("/", "_").replace("mtx", "dat"), shell=True)
	



