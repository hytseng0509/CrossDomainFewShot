from subprocess import call

call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/checkpoints/baseline.tar.gz', shell=True)
call('wget http://vllab.ucmerced.edu/ym41608/projects/CrossDomainFewShot/checkpoints/baseline++.tar.gz', shell=True)
call('tar -zxf baseline.tar.gz', shell=True)
call('tar -zxf baseline++.tar.gz', shell=True)
call('rm baseline.tar.gz', shell=True)
call('rm baseline++.tar.gz', shell=True)
