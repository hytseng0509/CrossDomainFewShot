from subprocess import call
import sys

# current available models:

if len(sys.argv) != 6:
  raise Exception('Incorrect command! e.g., python3 download_models.py [cub/cars/places/plantae] [1/5] [matchingnet/relationnet/gnnnet]')

testset = sys.argv[1]
shot = sys.argv[2]
model = sys.argv[3]

filename = 'multi_' + testset + '_' + shot + '_lft_' + model + '.tar.gz'

call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/checkpoints/' + filename, shell=True)
call('tar -zxf ' + filename, shell=True)
call('rm ' + filename, shell=True)
