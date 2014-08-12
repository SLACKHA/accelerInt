#comp plotter
with open("out.txt") as file:
	lines = file.readlines()[2:]
t1 = []
T1 = []
for line in lines:
	vals = line.strip().split("\t")
	if vals[0].startswith("Time:"):
		break
	t1.append(float(vals[0]))
	T1.append(float(vals[-1]))
	
with open("gpu_out.txt") as file:
	lines = file.readlines()[1:]
t3 = []
T3 = []
for line in lines:
	vals = line.strip().split("\t")
	if vals[0].startswith("Time:"):
		break
	t3.append(float(vals[0]))
	T3.append(float(vals[-1]))
	
with open("cantera_h2.txt") as file:
	lines = file.readlines()
t2 = []
T2 = []
for line in lines:
	vals = line.strip().split("\t")
	t2.append(float(vals[0]))
	T2.append(float(vals[-1]))
	
import matplotlib.pyplot as plt
fig = plt.figure()
plot = fig.add_subplot(1,1,1)

step = 4
t1_p = t1[0:len(t1):step]
T1_p = T1[0:len(T1):step]

t3_p = t3[2:len(t3):step]
T3_p = T3[2:len(T3):step]

plot.plot(t1_p, T1_p, marker = ">", linestyle = "", label = "exp4(cpu)")
plot.plot(t2, T2, label = "cantera")
plot.plot(t3_p, T3_p, marker = "v", linestyle = "", label = "exp4(gpu)")
plot.legend(loc = 0)
plot.set_ylabel("Temperature(K)")
plot.set_xlabel("Time (s)")
plt.show()

fig = plt.figure()
plot = fig.add_subplot(1,1,1)
T_diff = [100.0 * abs(a - b) / a for a, b in zip(T1, T3)]
plot.plot(t1, T_diff)
plot.set_ylabel("% Diff in Temperature (CPU:GPU)")
plot.set_xlabel("Time (s)")
plt.show()

fig = plt.figure()
plot = fig.add_subplot(1,1,1)
T_diff = [100.0 * abs(a - b) / a for a, b in zip(T2, T1)]
plot.plot(t1, T_diff)
plot.set_ylabel("% Diff in Temperature (CVODE:CPU)")
plot.set_xlabel("Time (s)")
plt.show()

fig = plt.figure()
plot = fig.add_subplot(1,1,1)
T_diff = [100.0 * abs(a - b) / a for a, b in zip(T2, T3)]
plot.plot(t1, T_diff)
plot.set_ylabel("% Diff in Temperature (CVODE:GPU)")
plot.set_xlabel("Time (s)")
plt.show()