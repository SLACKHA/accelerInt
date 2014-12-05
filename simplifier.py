from __future__ import division
from sympy import *
from sympy.printing import print_ccode as cprint
import re
import sys

def get_var_power_name(Base, power):
	if power == 1:
		return symbols(Base)
	return symbols(Base + ("inv" if power < 0 else "") + str(abs(power)))

def get_var_power(Var, power):
	if power == 0:
		return 1
	if power == 1:
		return Var
	if power < 0:
		ret_var = 1 / Var
		for i in range(1, abs(power)):
			ret_var /= Var
		return ret_var
	ret_var = Var
	for i in range(1, abs(power)):
		ret_var *= Var
	return ret_var


def opposite_comparison(comparator):
	if comparator == "<=":
		return ">"
	if comparator == "<":
		return ">="
	if comparator == ">":
		return "<="
	if comparator == ">=":
		return "<"
	raise Exception("Unknown comparator {}".format(comparator))

def write_powers(variable, outlines, powers):
	#find the first method
	location = None
	for i, line in enumerate(outlines):
		if "Real {}".format(variable) in line:
			location = i
			break

	#add the various powers
	minnegpow = None
	minnegpow_ind = None
	minpospow = None
	minpospow_ind = None
	for i, ipow in enumerate(powers):
		if ipow < 0:
			minnegpow = ipow
			minnegpow_ind = i
		elif minpospow is None and ipow > 1:
			minpospow = ipow
			minpospow_ind = i

	#start adding power lines
	prev_name = None
	for i in range(minnegpow_ind, -1, -1):
		if i == minnegpow_ind:
			#full define
			prev_name = str(get_var_power_name("T", powers[i]))
			outstr = "  Real " + prev_name + " = 1.0 / (" + " * ".join(["T" for j in range(abs(powers[i]))]) + ");\n"
			outlines.insert(location, outstr)
			location += 1
		else:
			outstr = "  Real " + str(get_var_power_name("T", powers[i])) + " = " + prev_name + " / (" + " * ".join(["T" for j in range(abs(powers[i] - powers[i + 1]))]) + ");\n"
			prev_name = str(get_var_power_name("T", powers[i]))
			outlines.insert(location, outstr)
			location += 1

	prev_name = "T"
	prev_pow = 1
	for i in range(minpospow_ind, len(powers)):
		outstr = "  Real " + str(get_var_power_name("T", powers[i])) + " = " + prev_name + " * (" + " * ".join(["T" for j in range(abs(powers[i] - prev_pow))]) + ");\n"
		prev_pow = powers[i]
		prev_name = str(get_var_power_name("T", powers[i]))
		outlines.insert(location, outstr)
		location += 1

	return outlines

class temperature_simplifier(object):
	def __init__(self, start_variable, match = None, line_end = ";\n"):
		self.T = symbols("T")
		self.match = match
		self.start_variable = start_variable
		self.line_end = line_end

	def get_clean_string(self, string, powers):
		expanded = expand(string)
		out_expr = None
		for i in range(-10, 11):
			coeff = expanded.coeff(self.T, i)
			if coeff:
				if out_expr is None:
					out_expr = (simplify(coeff)) * get_var_power_name("T", i) if i != 0 else (simplify(coeff))
				else:
					out_expr += (simplify(coeff)) * get_var_power_name("T", i) if i != 0 else (simplify(coeff))
				if i == 0 or i == 1:
					continue
				if not i in powers:
					powers.append(i)

		return str(out_expr)

	def reduce(self, lines):
		if self.match is None:
			raise Exception("Not implemented!")
		outlines = []
		powers = []
		for line in lines:
			thematch = re.search(self.match, line)
			if thematch:
				outlines.append(thematch.group(1) + self.get_clean_string(thematch.group(2), powers) + self.line_end)
			else:
				outlines.append(line)

		return outlines, powers

class repeated_simplifier(temperature_simplifier):
	def __init__(self, variable, array_list):
		temperature_simplifier.__init__(self, variable)
		self.var_match = re.compile("(" + variable + "\[(?:\w+|\d+)\])\s*([+-])?=\s*(.+);")
		self.array_match = [re.compile("(" + arr + ")\[(\w+|\d+)\]") for arr in array_list]
		self.back_array_match = [re.compile("(" + arr + ")\((\w+|\d+)\)") for arr in array_list]

	def construct(self, variable, expression, powers):
		ret_expr = []
		for expr in expression:
			temp = expr
			for array in self.array_match:
				temp = re.sub(array, r"\1(\2)", temp)
			ret_expr.append(temp)

		ret_expr = "+".join(["(" + str(expand(r)) + ")" for r in ret_expr])
		ret_expr = self.get_clean_string(ret_expr, powers)

		for array in self.back_array_match:
			ret_expr = re.sub(array, r"\1[\2]", ret_expr)

		return "  " + variable + " = " + ret_expr + ";\n"


	def reduce(self, lines):
		prev_expression = None
		prev_var = None
		powers = []
		outlines = []
		step = len(lines) / 10.0
		last_perc = 0
		for i, line in enumerate(lines):
			perc = (i / len(lines)) * 100
			if perc > last_perc + step:
				last_perc = perc
				print perc
			match = self.var_match.search(line)
			if match:
				variable = match.group(1)
				sign = match.group(2)
				if sign is None:
					sign = "+"
				expression = match.group(3)

				if prev_var != variable and prev_var is not None:
					outlines.append(self.construct(prev_var, prev_expression, powers))
					prev_var = None
				if prev_var is None:
					prev_var = variable
					prev_expression = ["{}(".format(sign if "-" in sign else "") + expression + ")"]
				else:
					prev_expression.append("{}(".format(sign) + expression + ")")
			else:
				if prev_var is not None:
					outlines.append(self.construct(prev_var, prev_expression, powers))
					prev_var = None
					prev_expression = None
				else:
					outlines.append(line)
		return outlines, powers

class if_statement_collapser(temperature_simplifier):
	def __init__(self, regex_list, lang = 'c'):
		temperature_simplifier.__init__(self, None)
		self.variables = [variable[0] for variable in regex_list]
		self.var_matchs = [re.compile("^(\\s*)(" + variable[0] + "\\s*)([+-])(=\\s*)(.+);\\s*$") for variable in regex_list]
		self.break_statements = [re.compile(regex[1]) for regex in regex_list] 
		self.if_statement_match = re.compile(r"if\s*\(T\s*([<>=]+)\s*([\d\.]+)")
		self.lang = lang

	def construct(self, variable, outlines, powers, stored):
		i = 0
		for key, value in stored.iteritems():
			if not len(value[0]):
				continue
			if self.lang == "c":
				outstr = "  if (T {} {}) {{\n".format(key[1], key[0])
				outlines.append(outstr)

				next_str = None
				if "<" in key[1]:
					next_str = " + ".join(value[1])
				else:
					next_str = " + ".join(value[0])

				outstr = self.get_clean_string(next_str, powers)
				outstr = "    {} {}= ".format(variable, "+" if i > 0 else "") + outstr + ";\n  } else {\n"
				outlines.append(outstr)

				if "<" in key[1]:
					next_str = " + ".join(value[0])
				else:
					next_str = " + ".join(value[1])
				
				outstr = self.get_clean_string(next_str, powers)
				outstr = "    {} {}= ".format(variable, "+" if i > 0 else "") + outstr + ";\n"
				outstr += "  }\n"
				outlines.append(outstr)
				i += 1
			elif self.lang == "cuda":
				outstr = " (T {} {}) * ".format(key[1], key[0])

				next_str = None
				if "<" in key[1]:
					next_str = " + ".join(value[1])
				else:
					next_str = " + ".join(value[0])
				outstr += "(" + self.get_clean_string(next_str, powers) + ")"
				
				outstr +="+ (T {} {}) * ".format(opposite_comparison(key[1]), key[0])
				if "<" in key[1]:
					next_str = " + ".join(value[0])
				else:
					next_str = " + ".join(value[1])

				outstr += self.get_clean_string(next_str, powers)
				
				outstr += ";\n"
				outlines.append(outstr)
			else:
				raise Exception("unknown lang")

	def __check_var_match(self, line):
		for i, varm in enumerate(self.var_matchs):
			match = varm.search(line)
			if match:
				return i, match
		return -1, None

	def __check_break_match(self, line):
		for i, breakm in enumerate(self.break_statements):
			match = breakm.search(line)
			if match:
				return i, match
		return -1, None

	def reduce(self, lines):
		outlines = []
		powers = []

		stored = {}
		for variable in self.variables:
			stored[variable] = {}
		for index, line in enumerate(lines):
			#temperature if statement
			processed = False
			match = self.if_statement_match.search(line)
			processed = processed or match
			if match:
				#check that next line matches our variable
				ind, next_match = self.__check_var_match(lines[index + 1])
				if next_match is None:
					processed = False
				else:
					T_mid = float(match.group(2))
					operator = match.group(1)
					if not (T_mid, operator) in stored[self.variables[ind]]:
						stored[self.variables[ind]][(T_mid, operator)] = ([], [])

					next_low = "<" in match.group(2)

			#var match
			ind, match = self.__check_var_match(line)
			processed = processed or match
			if match:
				clean = match.group(5).strip()
				the_string = ""
				if next_low:
					if len(stored[self.variables[ind]][(T_mid, operator)][0]) or "-" in match.group(3):
						the_string += match.group(3)
					the_string += " (" + clean + ") "
					stored[self.variables[ind]][(T_mid, operator)][0].append(the_string)
				else:
					if len(stored[self.variables[ind]][(T_mid, operator)][1]) or "-" in match.group(3):
						the_string += match.group(3)
					the_string += " (" + clean + ") "
					stored[self.variables[ind]][(T_mid, operator)][1].append(the_string)
				next_low = not next_low

			#break of temperature if's
			ind, match = self.__check_break_match(line)
			processed = processed or match
			if match:
				self.construct(self.variables[ind], outlines, powers, stored[self.variables[ind]])
				outlines.append(line)
				stored[self.variables[ind]] = {}

		
			if not processed:
				if not any(stored[val] for val in stored):
					outlines.append(line)

		return outlines, powers


# with open("src/methane/rxn_rates.c") as infile:
# 	lines = infile.readlines()

# c = temperature_simplifier("Kc", "(\\s*kf\\s*=\\s*exp\\()(.*\\bT.*)\\);", ");\n")
# outlines, powers = c.reduce(lines)
# c = if_statement_collapser(["Kc", "exp\\(Kc\\)"])
# outlines, p1 = c.reduce(outlines)
# powers = sorted(list(set(powers + p1)))
# outlines = write_powers("Kc", outlines, powers)

# with open("src/methane/new_rxn_rates.c", "w+") as outfile:
# 	outfile.writelines(outlines)


with open("src/methane/jacob.c") as infile:
	lines = infile.readlines()

#c = repeated_simplifier("jac", ["fwd_rxn_rates", "rev_rxn_rates", "pres_mod", "conc", "\\by", "sp_rates"])
#outlines, powers = c.reduce(lines)
c = if_statement_collapser([("Kc", "exp\\(Kc\\)"), ("jac[0]", "jac[0]\s*\*=")])
outlines, powers = c.reduce(lines)
with open("src/methane/new_jacob.c", "w+") as outfile:
	outfile.writelines(outlines)
sys.exit(1)
c = if_statement_collapser("jac[0]", "jac[0]\s*\*=")
outlines, p1 = c.reduce(outlines)
powers = sorted(list(set(powers + p1)))
outlines = write_powers("mw_avg", outlines, powers)
with open("src/methane/new_jacob.c", "w+") as outfile:
	outfile.writelines(outlines)