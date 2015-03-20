import cantera as ct
import numpy as np

t_end = 0.001
def main():
    gas = ct.Solution("h2.xml")
    with open("ign_data.txt", "w+") as file:
        #conditions
        T_start = np.arange(1000.0, 2500.0, 25.0)
        P_start = np.arange(101325.0, 50.0 * 101325.0, 5.0 * 101325.0)
        Phi_start = np.arange(1.0, 3.0, 1.0)

        my_sum = 0

        runs = len(T_start) * len(P_start) * len(Phi_start)
        run = 0
        for T in T_start:
            for P in P_start:
                for Phi in Phi_start:
                    print "Run: {:} of {:}".format(run + 1, runs)
                    run += 1 
                    my_sum += run_sim(T, P, Phi, file, gas)
        print "# of Conditions = " + str(my_sum)

def run_sim(T, P, Phi, file, gas):
    #state array for each simulation state, and each mass fraction + T & P
    state = np.zeros(gas.n_species + 2)
    states = 0

    #constant pressure run
    gas.TPX = T,P, "H2:" + str(Phi) + ", O2:1.0, N2:3.76"
    reac = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reac])

    t = 0
    while t < t_end:
        state[0] = reac.T
        state[1] = reac.thermo.P 
        state[2:] = reac.Y[:]
        file.write(' '.join(np.char.mod('%f', state[:])) + "\n")
        t = net.step(t_end)
        states += 1

    #conv run
    gas.TPX = T,P, "H2:" + str(Phi) + ", O2:1.0, N2:3.76"
    reac = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reac])

    while t < t_end:
        state[0] = reac.T
        state[1] = reac.thermo.P 
        state[2:] = reac.Y[:]
        state_array[i, :] = state[:]
        file.write(' '.join(np.char.mod('%f', state[:])) + "\n")
        t = net.step(t_end)
        states += 1

    return states
                    


if __name__ == "__main__":
   main()