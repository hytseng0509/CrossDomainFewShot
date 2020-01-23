from subprocess import call
import sys

# current available models:
# multi cars 1 ori gnnnet

if len(sys.argv) != 6:
  raise Exception('Incorrect command! e.g., python3 download_models.py [single/multi] [miniImagenet/cub/cars/places/plantae] [1/5] [ori/fa/lft] [matchingnet/relationnet/gnnnet]')

task = sys.argv[1]
testset = sys.argv[2]
shot = sys.argv[3]
method = sys.argv[4]
model = sys.argv[5]

filename = task + '_' + testset + '_' + shot + '_' + method + '_' + model + '.tar.gz'

call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/checkpoints/' + filename, shell=True)
call('tar -zxf ' + filename, shell=True)
call('rm ' + filename, shell=True)
