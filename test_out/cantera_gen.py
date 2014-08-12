import cantera as ct
gas = ct.Solution("h2.cti")
gas.TPX = 1600,101325,"H2:2.0, O2:1.0, N2:3.76"
reac = ct.IdealGasConstPressureReactor(gas)
net = ct.ReactorNet([reac])

t = 0
dt = 1e-6
t_end = 1e-3

with open("cantera_h2.txt", "w+") as file:
	while t < t_end:
		net.advance(t + dt)
		t += dt
		file.write(str(t) + "\t" + str(reac.T) + "\n")