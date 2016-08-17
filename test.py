from subprocess import call
from sys import argv
import os

path = "./"
files = []
script, filename = argv
txt = open(filename)

mm_path = "/home/crtrott/matrices/MM/"

print "Generating results for all matrices in  %r:" % filename

for filename in txt:
 	call("srun ./test_matvec.cuda-new -fb " + mm_path + filename, shell=True),
	for i in os.listdir(path):
    		if os.path.isfile(os.path.join(path,i)) and 'k40' in i:
			call("mv " + i + " " + filename.replace("/", "_").replace("mtx", "dat"), shell=True)
	



