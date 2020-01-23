from subprocess import call
import sys

if len(sys.argv) != 5:
  raise Exception('Incorrect command! e.g., python3 process.py [single/multi] [miniImagenet/cub/cars/places/plantae] [ori/fa/lft] [matchingnet/relationnet/gnnnet]')

task = sys.argv[1]
testset = sys.argv[2]
method = sys.argv[3]
model = sys.argv[4]

filename = task + '_' + testset + '_' + method + '_' + model + '.tar.gz'

call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/checkpoints/' + filename, shell=True)
call('tar -zxf ' + filename, shell=True)
call('rm ' + filename, shell=True)
