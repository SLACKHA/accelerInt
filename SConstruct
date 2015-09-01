#! /usr/env/bin python2.7
import os
import re

env = Enviroment()

def get_files(directory, extension, file_filter=None, inverse_filter=None):
	file_list = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
	 and f.endswith(extension)]
	if file_filter:
		if isinstance(file_filter, list):
			file_list = [f for f in file_list if all(re.search(filt, f) for filt in file_filter)]
		else:
			file_list = [f for f in file_list if re.search(file_filter, f)]
	if inverse_filter:
		if isinstance(inverse_filter, list):
			file_list = [f for f in file_list if not any(re.search(filt, f) for filt in inverse_filter)]
		else:
			file_list = [f for f in file_list if not re.search(inverse_filter, f)]
	return file_list

#directories
home = os.getcwd()
mech_dir = os.path.join(home, 'mechanism', 'src')
generic_dir = os.path.join(home, 'generic', 'src')
radau2a_dir = os.path.join(home, 'radau2a', 'src')
exp_int_dir = os.path.join(home, 'exponential_integrators', 'src')
exp4_int_dir = os.path.join(home, 'exponential_integrators', 'exp4', 'src')
exprb43_int_dir = os.path.join(home, 'exponential_integrators', 'exprb43', 'src')
cvodes_dir = os.path.join(home, 'cvodes', 'src')

#common file lists
mechanism_src = get_files(mech_dir, '.c')
mechanism_cuda_src = get_files(mech_dir, '.cu') + get_files(mech_dir, '.c', file_filter='mass_mole')

generic_src = get_files(generic_dir, '.c', inverse_filter='fd_jacob')
generic_cuda_src = get_files(generic_dir, '.cu', inverse_filter='fd_jacob')

solver_and_mech = mechanism_src + generic_src
solver_and_mech_cuda = mechanism_cuda_src + generic_cuda_src

exp_int_src = get_files(exp_int_dir, '.c')
exp_int_cuda_src = get_files(exp_int_dir, '.cu') + get_files(exp_int_dir, '.c', file_filter='linear-algebra')

exp_solver_and_mech = solver_and_mech + exp_int_src
exp_solver_and_mech_cuda = solver_and_mech + exp_int_cuda_src

cvodes_base_src = filter(lambda x: not 'jacob' in x, mechanism_src) + get_files(generic_dir, '.c', inverse_filter='solver_generic')
cvodes_analytical_src = mechanism_src + get_files(generic_dir, '.c', inverse_filter='solver_generic')

#set up targets
target_list = []
terget_list.append(
	env.Program(target='radau2a-int', sources=solver_and_mech + get_files(radau2a_dir, '.c')]))
target_list.append(
	env.Program(target='radau2a-int-gpu', sources=solver_and_mech_cuda + get_files(radau2a_dir, '.cu')]))
target_list.append(
	env.Program(target='exp4-int', sources=exp_solver_and_mech + get_files(exp4_int_dir, '.c')]))
target_list.append(
	env.Program(target='exp4-int-gpu', sources=exp_solver_and_mech_cuda + get_files(exp4_int_dir, '.cu')]))
target_list.append(
	env.Program(target='exprb4-int', sources=exp_solver_and_mech + get_files(exprb4_int_dir, '.c')]))
target_list.append(
	env.Program(target='exprb4-int-gpu', sources=exp_solver_and_mech_cuda + get_files(exprb4_int_dir, '.cu')]))
target_list.append(
	env.Program(target='cvodes-int', sources=cvodes_base_src + get_files(cvodes_dir, '.c', inverse_filter='cvodes_jac')]))
target_list.append(
	env.Program(target='cvodes-analytical-int', sources=cvodes_base_src + get_files(cvodes_dir, '.c', inverse_filter='cvodes_jac')]))
