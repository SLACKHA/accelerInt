#! /usr/bin/env python2.7

import cantera as ct
import numpy as np
import random
import multiprocessing
from argparse import ArgumentParser

t_step = 1e-6
t_end = 0.001
N_MAX = 1048576 #2^20
def compute_stoich(gas, fuel, Phi):
    try:
        C_mol = gas.n_atoms(fuel, 'C')
    except:
        C_mol = gas.n_atoms(fuel, 'c')

    try:
        H_mol = gas.n_atoms(fuel, 'H')
    except:
        H_mol = gas.n_atoms(fuel, 'h')

    try:
        O_mol = gas.n_atoms(fuel, 'O')
    except:
        O_mol = gas.n_atoms(fuel, 'o')

    O2_name = "O2" if "O2" in gas.species_names else "o2"
    N2_name = "N2" if "N2" in gas.species_names else "n2"

    #compute stoichoimetry
    #Phi * fuel + y * (O2 + 3.76N2) = C_mol * Phi * CO2 + z*H20 + y * 3.76N2

    #O sums
    #Phi * O_mol + 2y = C_mol * Phi * 2 + z

    #H sums
    #Phi * H_mol = 2z

    z = Phi * H_mol / 2.0

    y = (C_mol * Phi * 2.0 + z - Phi * O_mol) / 2.0

    return "{}:{}, {}:{}, {}:{}".format(fuel, Phi, O2_name, y, N2_name, 3.76 * y)

def write_array(file, array):
    if array is None:
        return 0
    for i in range(array.shape[0]):
        file.write(' '.join(np.char.mod('%.15e', array[i, :])) + "\n")
    return array.shape[0]


def main(mechanism, fuel, n_threads=12):
    gas = ct.Solution(mechanism)
    if not fuel in gas.species_names:
        raise Exception("Fuel not found!")

    mechanism = mechanism.replace(".cti", ".xml")
    file = open("ign_data.txt", "w")
    #conditions
    T_start = np.linspace(1000.0, 3000.0, num=25)
    P_start = np.linspace(101325.0, 50.0 * 101325.0, num=25)
    Phi_start = np.arange(0.25, 3.0, 0.25)

    my_sum = 0

    pool = multiprocessing.Pool(n_threads)

    runs = len(T_start) * len(P_start) * len(Phi_start)
    run = 0
    total = 0
    results = []
    for T in T_start:
        for Phi in Phi_start:
            for P in P_start:
                results.append(pool.apply_async(run_sim, [run, mechanism, fuel, T, P, Phi]))
                run += 1
        arrays = [p.get() for p in results]
        for i in range(len(arrays)):
            my_sum += write_array(file, arrays[i][1])
            my_sum += write_array(file, arrays[i][2])
        total += len(arrays)
        print "{} / {}".format(total, runs)

        run = 0
        results = []
    print "# of Conditions = " + str(my_sum) + (" > " if my_sum > N_MAX else " < ") + '2 ^ 20'
    shuffle_states()
    file.flush()
    file.close()

def run_sim(index, mech, fuel, T, P, Phi):
    gas = ct.Solution(mech)
    gas.TPX = T, P, compute_stoich(gas, fuel, Phi)
    #state array for each simulation state, and each mass fraction + T & P
    states = np.zeros((t_end / t_step, gas.n_species + 2))
    conp_states = None
    conv_states = None

    reac = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reac])

    t = 0
    i = 0
    while t < t_end:
        states[i, 0] = reac.T
        states[i, 1] = reac.thermo.P 
        states[i, 2:] = reac.Y[:]
        net.advance(t + t_step)
        t += t_step
        i += 1

    if reac.T > T + 400:
        #ignition, use it
        conp_states = np.copy(states)

    #conv run
    gas.TPX = T, P, compute_stoich(gas, fuel, Phi)
    #state array for each simulation state, and each mass fraction + T & P
    states = np.zeros((t_end / t_step, gas.n_species + 2))
    reac = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reac])

    t = 0
    i = 0
    while t < t_end:
        states[i, 0] = reac.T
        states[i, 1] = reac.thermo.P 
        states[i, 2:] = reac.Y[:]
        net.advance(t + t_step)
        t += t_step
        i += 1
    if reac.T > T + 400:
        #ignition, use it
        conv_states = np.copy(states)

    return index, conp_states, conv_states

def shuffle_states():
    with open("ign_data.txt") as file:
        lines = [line.strip() for line in file]
    random.shuffle(lines)
    with open("shuffled_data.txt", "w") as file:
        file.write("\n".join(lines))

if __name__ == "__main__":
    parser = ArgumentParser(description='Generates initial conditions for a specified fuel / mechanism')
    parser.add_argument('-f', '--fuel',
                    type=str,
                    required=True,
                    help = 'The fuel molecule')
    parser.add_argument('-m', '--mechanism',
                    type=str,
                    required=True,
                    help = 'The mechanism to use')
    args = parser.parse_args()
    main(args.mechanism, args.fuel)