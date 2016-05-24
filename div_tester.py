#meant to be run after runner.py
#does not craete directories or generate mechanisms or any of that
#just a quick and dirty tester

import subprocess
import shutil
import os
from glob import glob

DIVERGENCE_WARPS = 8192


mechs = ['H2', 'CH4']
dts = [1e-6, 1e-4]
solvers = ['radau2a-int-gpu', 'exprb43-int-gpu', 'exp4-int-gpu']

for mech in mechs:
    for dt in dts:
        shutil.copyfile(os.path.join(os.getcwd(), 'performance', mech, 'data_eqremoved.bin'), 
            os.path.join(os.getcwd(), 'ign_data.bin'))
        thedir = 'performance/{}/gpu_nco_nosmem'.format(mech)
        subprocess.check_call(['scons', 'gpu', 'mechanism_dir={}'.format(thedir), '-j12', 'DIVERGENCE_WARPS={}'.format(DIVERGENCE_WARPS),
            't_step={:.0e}'.format(dt), 't_end={:.0e}'.format(dt)])
        for solver in solvers:
            subprocess.check_call(os.path.join(os.getcwd(), solver))

        for f in glob('log/*div.txt'):
            shutil.copyfile(f, os.path.join(os.getcwd(), 'divergence', mech, '{:.0e}'.format(dt),
                f[f.rindex('/') + 1:]))

