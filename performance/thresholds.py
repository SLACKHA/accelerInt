"""
thresholds.py

Contains a list of thresholds for each performance case
after which the solver cost/ode is roughly constant
-------------------------------------------------------

Used for plotting and GPU/CPU comparison plots
"""

thresholds = {
    'H2' : {
        'gpu' : {
            1e-6 : 4096,
            1e-4 : 4096
        },
        'cpu' : {
            1e-6 : 32768,
            1e-4 : 16384
        }
    },
    'CH4' : {
        'gpu' : {
            1e-6 : 2048,
            1e-4 : 4096
        },
        'cpu' : {
            1e-6 : 4096,
            1e-4 : 2048
        }
    }
}