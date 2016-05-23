"""
thresholds.py

Contains a list of thresholds for each performance case
after which the solver cost/ode is roughly constant
-------------------------------------------------------

Used for plotting and GPU/CPU comparison plots
"""

import sys
thresholds = None

def parse_file():
    t = {}
    try:
        with open('thresholds.txt', 'r') as file:
            for line in file:
                mech, gpu, dt, xval = line.strip().split('\t')
                dt = float(dt)
                gpu = True if gpu == 'gpu' else False
                t[(mech, gpu, dt)] = int(float(xval))
    except Exception, e:
        print(e)
        print('Run nominal_performance.py first')
        sys.exit(-1)
    return t

def get_threshold(mech, gpu, dt):
    global thresholds
    if thresholds is None:
        thresholds = parse_file()
    return thresholds[(mech, gpu, dt)]