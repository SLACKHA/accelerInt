import cantera as ct
import numpy as np
import multiprocessing as mp

def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open("ign_data.txt", "w+") as file:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            file.write(m + '\n')
            file.flush()

samples = 200.0
t_end = 100
size = int(1e6)
def main():
    manager = mp.Manager()
    q = manager.Queue()
    THREADS = 12
    pool = mp.Pool(THREADS + 1)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #conditions
    T_start = np.arange(760.0, 2500.0, 50.0)
    P_start = np.arange(101325.0, 50.0 * 101325.0, 5.0 * 101325.0)
    Phi_start = np.arange(0.5, 3.5, 0.1)

    jobs = []
    for T in T_start:
        for P in P_start:
            for Phi in Phi_start:
                job = pool.apply_async(run_sim, (T, P, Phi, q))
                jobs.append(job)

    # collect results from the workers through the pool result queue
    my_sum = 0
    for i, job in enumerate(jobs):
        my_sum += job.get()
        print "Job #" + str(i) + " of " + str(len(jobs)) + " complete."
    print "# of Conditions = " + str(my_sum * samples)

    #now we are done, kill the listener
    q.put('kill')
    pool.close()

def run_sim(T, P, Phi, q):
    gas = ct.Solution("h2.xml")
    t = 0
    ignitions = 0

    #state array for each simulation state, and each mass fraction + T & P
    state_array = np.zeros((size, gas.n_species + 2))
    state = np.zeros(gas.n_species + 2)

    #constant pressure run
    gas.TPX = T,P, "H2:" + str(Phi) + ", O2:1.0, N2:3.76"
    reac = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reac])

    i = 0
    t = 0
    while t < t_end and i < size:
        state[0] = reac.T
        state[1] = reac.thermo.P 
        state[2:] = reac.Y[:]
        state_array[i, :] = state[:]
        t = net.step(t_end)
        i += 1

    #ok, now get samples based on equidistant T
    T_0 = state_array[0, 0]
    T_end = state_array[i - 1, 0]
    if T_end > T_0 + 400:
        ignitions += 1
        delta_T = (T_end - T_0) / samples
        T_next = T_0
        for i in range(size):
            if state_array[i, 0] >= T_next:
                arr_str = np.char.mod('%f', state_array[i, :])
                q.put(' '.join(arr_str))
                T_next += delta_T
            if T_next > T_end:
                break

    #conv run
    gas.TPX = T,P, "H2:" + str(Phi) + ", O2:1.0, N2:3.76"
    reac = ct.IdealGasReactor(gas)
    net = ct.ReactorNet([reac])

    i = 0
    t = 0
    while t < t_end and i < size:
        state[0] = reac.T
        state[1] = reac.thermo.P 
        state[2:] = reac.Y[:]
        state_array[i, :] = state[:]
        t = net.step(t_end)
        i += 1

    #ok, now get samples based on equidistant T
    T_0 = state_array[0, 0]
    T_end = state_array[i - 1, 0]
    if T_end > T_0 + 400:
        ignitions += 1
        delta_T = (T_end - T_0) / samples
        T_next = T_0
        for i in range(size):
            if state_array[i, 0] >= T_next:
                arr_str = np.char.mod('%f', state_array[i, :])
                q.put(' '.join(arr_str))
                T_next += delta_T
            if T_next > T_end:
                break
    return ignitions
                    


if __name__ == "__main__":
   main()