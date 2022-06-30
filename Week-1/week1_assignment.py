"""

Name		: Sai Shashank GP
Roll no		: EE20B040
Course		: EE2703 - Applied Programming Lab
Assignment 	: 1
Date		: 19-01-2022
Description	: Read a .netlist file, reverse the each component string and print each component string
Language	: Python3

"""

# Importing any useful librares or modules
import sys

# 1. Accept the netlist file as a commandline argument 

'''
Possible hurdles:
1. More than file name given
2. Wrong file name or file path given
'''

# 2. Parse the file starting from '.circuit' to '.end'

'''
Possible hurdles:
1. Each line ends with "\n"
Approach:
1. Introducing "\n" to START & STOP variables 
2. Stripping out "\n" in all cases from now on 
'''

START = ".circuit"
STOP = ".end"

def parseCkt(file):
	'''
	This function parses the circuit from .circuit to .end and returns the ckt component strings list.
	It checks if the file is malformed. Like, by missing any mandatory elements like .circuit or .end
	'''
	contents = list(file.readlines())
	contents = [list(i.split('#'))[0].strip() for i in contents]
	try:
		indStart = contents.index(START)
		indStop = contents.index(STOP)
	except ValueError:
		print("ERROR: File is malformed! There is no circuit in the file")
		sys.exit()
	if indStart > indStop:
		print("ERROR: File is malformed! Please check the syntax of file")
		sys.exit()
	ckt = contents[indStart+1:indStop]
	try:
		assert len(ckt) == len(set(ckt))
	except AssertionError:
		print("ERROR: File is malformed! Please check the circuit to remove repititions")
		sys.exit()
	try:
		assert START not in ckt
		assert STOP not in ckt
	except AssertionError:
		print("WARNING: Please check no.of '.circuit' & '.end' in the file")
	file.close()
	return ckt

# 3. Parse each component string and extract the component information

'''
Possible hurdles:
1. Ignoring comments(comments are represented by "#")
Approach:
Use string.split() method with "#" as argument so that other part might be ignored
'''

def parseComp(ckt):
	'''This function parses the circuit list and breaks each component string into individual words'''
	components = []
	for line in ckt:
		comp = list(line.split('#'))[0].strip()
		components.append(comp.split())
	return components

# # 4. Print the reversed component string

def printReversed(components):
	'''This function prints the reversed component string in a circuit'''
	for comp in components[::-1]:
		comp = comp[::-1]
		print(' '.join(comp).strip())


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("ERROR: Please enter valid no. of arguments. Acceptable no.of arguments are 2")
	else:
		try:
			cktfilename = sys.argv[1].strip()
		except FileNotFoundError:
			print("ERROR: Please enter a valid file name or file path")
			sys.exit()
		cktfile = open(cktfilename)
		ckt = parseCkt(cktfile)
		components = parseComp(ckt)
		printReversed(components)
