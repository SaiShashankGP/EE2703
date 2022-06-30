"""

Name        : Sai Shashank GP
Roll no     : EE20B040
Course      : EE2703
Assignment  : 2
Date        : 09-02-2022
Description : Solve the circuits using scientific python modules(numpy) and OOP
Language    : Python3

"""

# Importing Useful Libraries
import sys
import numpy as np

# initialising all the required variables
global j, w, circuit, Nodes, isAC
VoltageSources = []
j = 0 + 1j
START = ".circuit"
STOP = ".end"
AC = ".ac"
circuit = []
Nodes = []

def parseCkt(filename):
    '''
    This function parses the netlist file and finds the key arguments like .circuit etc... 
    It also validates the netlist file and raises error if needed
    '''
    file = open(filename)
    contents = list(file.readlines())
    contents = [list(i.split('#'))[0].strip() for i in contents]
    try:
        indStart = contents.index(START)
        indStop = contents.index(STOP)
    except ValueError:
        print("ERROR: File is malformed! There is no circuit in the file")
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
    acfreqs = []
    for line in contents[indStop:]:
        if line[:3] == AC:
            acfreqs.append(line)
    return ckt, acfreqs

# 3. Parse each component string and extract the component information

def parseComp(ckt):
	'''This function parses the circuit list and breaks each component string into individual words'''
	components = []
	for line in ckt:
		components.append(line.split())
	return components

# 5. Check for AC elements in the ciruit

def checkAC(file):
	'''This function checks for AC elements in the circuits'''
	lines = file.readlines()
	for line in lines:
		if list(line.split('#'))[0].strip().split()[0] == AC:
			return True
	return False

def findPhase(a):
    '''
    It returns the phase of a complex number in degrees without running into errors
    NOTE: It uses numpy module to work
    '''
    if a == 0j:
        return 0.0
    else:
        return np.arctan(a.imag/a.real)*180/np.pi

def solveDC():
    '''This function solves DC circuits which doesn't require transient analysis'''
    len_ = len(Nodes) + len(VoltageSources)
    M = np.zeros((len_,len_))
    B = np.zeros((len_, ))
    for elem in circuit:
        elem.updateDCMat(M, B)
    x = np.linalg.solve(M, B)
    x_v, x_i = x[:len(Nodes)], x[len(Nodes):]
    for i in range(len(x_v)):
        print(f"Voltage at {Nodes[i]}: {x_v[i]} V")
    for i in range(len(x_i)):
        print(f"Current thorugh Voltage Source {VoltageSources[i]}: {x_i[i]} A")

def solveAC():
    '''This function solves AC circuits which doesn't require transient analysis'''
    len_ = len(Nodes) + len(VoltageSources)
    M = np.zeros((len_,len_), dtype=complex)
    B = np.zeros((len_, ), dtype=complex)
    for elem in circuit:
        elem.updateACMat(M, B)
    x = np.linalg.solve(M, B)
    print(x)
    x_v, x_i = x[:len(Nodes)], x[len(Nodes):]
    for i in range(len(x_v)):
        print(f"Voltage at {Nodes[i]}:\t{2*abs(x_v[i])} Vp-p with Phase {findPhase(x_v[i])} degrees")
    for i in range(len(x_i)):
        print(f"Current thorugh Voltage Source {VoltageSources[i]}:\t{2*abs(x_i[i])} Ap-p with Phase {findPhase(x_i[i])} degrees")

# Defining some classes of elements as required

class TwoNodeElement:
    '''
    This class is used to represent Two-node elements in the circuit\n
    It has following attributes\n
        1. Name\n
        2. From node\n
        3. To node\n
        4. Value\n
    NOTE: from node and to node doesn't make sense with bilateral components like resistors.\nBut it is considered due to Passive sign Convention
    '''

    def __init__(self, componentattrs: list):
        self.componentattrs = componentattrs
        self.compName = self.componentattrs[0]
        self.compValue = float(self.componentattrs[-1])
        self.fromNode = self.componentattrs[1]
        self.toNode = self.componentattrs[2]
        if self.fromNode.lower() == 'gnd':
            self.fromNode = 'GND'
        if self.toNode.lower() == 'gnd':
            self.toNode = 'GND'

class Resistor(TwoNodeElement):
    '''
    This class is used to represent Resistor in the circuit\n
    This class inherits from TwoNodeElement class
    '''

    def __init__(self, componentattrs: list):
        self.compType = 'Resistor'
        TwoNodeElement.__init__(self, componentattrs)
        if self.fromNode not in Nodes:
            Nodes.append(self.fromNode)
        if self.toNode not in Nodes:
            Nodes.append(self.toNode)
        
    def updateDCMat(self, M, B):
        '''Updates the M & B matrices according to the Resistor'''
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[fromNodeInd][fromNodeInd] = 1
            M[toNodeInd][toNodeInd] += 1/self.compValue
            M[toNodeInd][fromNodeInd] -= 1/self.compValue
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[toNodeInd][toNodeInd] = 1
            M[fromNodeInd][fromNodeInd] += 1/self.compValue
            M[fromNodeInd][toNodeInd] -= 1/self.compValue
            return
        M[fromNodeInd][fromNodeInd] += 1/self.compValue
        M[toNodeInd][toNodeInd] += 1/self.compValue
        M[fromNodeInd][toNodeInd] -= 1/self.compValue
        M[toNodeInd][fromNodeInd] -= 1/self.compValue
    
    def updateACMat(self, M, B):
        '''This function updates the M & B matrices according to the Resistor'''
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[fromNodeInd][fromNodeInd] = 1 + 0j
            M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
            M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[toNodeInd][toNodeInd] = 1 + 0j
            M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
            M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
            return
        M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
        M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
        M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
        M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j

class Inductor(TwoNodeElement):
    '''
    This class is used to represent Inductor in the circuit\n
    This class inherits from TwoNodeElement class
    '''

    def __init__(self, componentattrs: list):
        self.compType = 'Inductor'
        TwoNodeElement.__init__(self, componentattrs)
        if self.fromNode not in Nodes:
            Nodes.append(self.fromNode)
        if self.toNode not in Nodes:
            Nodes.append(self.toNode)
    
    def updateDCMat(self, M, B):
        '''This function is defined incase user inputs Inductor in DC circuits'''
        print("ERROR: Inductor cannot be used in non-transient analysis of DC circuits")
        sys.exit()
    
    def updateACMat(self, M, B):
        '''This function updates the M & B matrices according to the Inductor'''
        self.compValue = (w*self.compValue)*j
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[fromNodeInd][fromNodeInd] = 1 + 0j
            M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
            M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[toNodeInd][toNodeInd] = 1 + 0j
            M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
            M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
            return
        M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
        M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
        M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
        M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j

class Capacitor(TwoNodeElement):
    '''
    This class is used to represent Inductor in the circuit\n
    This class inherits from TwoNodeElement class
    '''

    def __init__(self, componentattrs: list):
        self.compType = 'Capacitor'
        TwoNodeElement.__init__(self, componentattrs)
        if self.fromNode not in Nodes:
            Nodes.append(self.fromNode)
        if self.toNode not in Nodes:
            Nodes.append(self.toNode)
    
    def updateDCMat(self, M, B):
        '''This function is defined incase user inputs Capacitor in DC circuits'''
        print("ERROR: Capacitor cannot be used in non-transient analysis of DC circuits")
        sys.exit()
    
    def updateACMat(self, M, B):
        '''This function updates the M & B matrices according to the Capacitor'''
        self.compValue = -j/(w*self.compValue)
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[fromNodeInd][fromNodeInd] = 1 + 0j
            M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
            M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[toNodeInd][toNodeInd] = 1 + 0j
            M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
            M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
            return
        M[fromNodeInd][fromNodeInd] += 1/self.compValue + 0j
        M[toNodeInd][toNodeInd] += 1/self.compValue + 0j
        M[fromNodeInd][toNodeInd] -= 1/self.compValue + 0j
        M[toNodeInd][fromNodeInd] -= 1/self.compValue + 0j


class VoltageSource(TwoNodeElement):
    '''
    This class is used to represent Independent Voltage Source in the circuit\n
    This class inherits from TwoNodeElement class
    '''

    def __init__(self, componentattrs: list):
        self.compType = 'VoltageSource'
        self.phase = 0
        self.frequency = 0
        self.type = 'dc'
        TwoNodeElement.__init__(self, componentattrs)
        if self.fromNode not in Nodes:
            Nodes.append(self.fromNode)
        if self.toNode not in Nodes:
            Nodes.append(self.toNode)
    
    def updateDCMat(self, M, B):
        '''Updates the M & B matrices according to the Voltage source'''
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        sourceInd = VoltageSources.index(self.compName)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[toNodeInd][len(Nodes)+sourceInd] = -1
            M[len(Nodes)+sourceInd][fromNodeInd] = -1
            M[len(Nodes)+sourceInd][toNodeInd] = 1 
            B[len(Nodes)+sourceInd] = self.compValue
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[fromNodeInd][len(Nodes)+sourceInd] = 1
            M[len(Nodes)+sourceInd][fromNodeInd] = -1
            M[len(Nodes)+sourceInd][toNodeInd] = 1 
            B[len(Nodes)+sourceInd] = self.compValue
            return
        M[fromNodeInd][fromNodeInd] = 1
        M[toNodeInd][toNodeInd] = -1
        M[len(Nodes)+sourceInd][fromNodeInd] = -1
        M[len(Nodes)+sourceInd][toNodeInd] = 1 
        B[len(Nodes)+sourceInd] = self.compValue
    
    def updateACMat(self, M, B):
        '''Updates the M & B matrices according to the Voltage source'''
        self.compValue = self.compValue/2
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        sourceInd = VoltageSources.index(self.compName)
        try:
            assert self.fromNode != 'GND'
        except AssertionError:
            M[toNodeInd][len(Nodes)+sourceInd] = -1 + 0j
            M[len(Nodes)+sourceInd][fromNodeInd] = -1 + 0j
            M[len(Nodes)+sourceInd][toNodeInd] = 1 + 0j
            B[len(Nodes)+sourceInd] = self.compValue + 0j
            return
        try:
            assert self.toNode != 'GND'
        except AssertionError:
            M[fromNodeInd][len(Nodes)+sourceInd] = 1 + 0j
            M[len(Nodes)+sourceInd][fromNodeInd] = -1 + 0j
            M[len(Nodes)+sourceInd][toNodeInd] = 1 + 0j
            B[len(Nodes)+sourceInd] = self.compValue + 0j
            return
        M[fromNodeInd][fromNodeInd] = 1 + 0j
        M[toNodeInd][toNodeInd] = -1 + 0j
        M[len(Nodes)+sourceInd][fromNodeInd] = -1 + 0j
        M[len(Nodes)+sourceInd][toNodeInd] = 1 + 0j
        B[len(Nodes)+sourceInd] = self.compValue + 0j

class CurrentSource(TwoNodeElement):
    '''
    This class is used to represent Independent Current Source in the circuit\n
    This class inherits from TwoNodeElement class
    '''

    def __init__(self, componentattrs: list):
        self.compType = 'CurrentSource'
        self.phase = 0
        self.frequency = 0
        self.type = 'dc'
        TwoNodeElement.__init__(self, componentattrs)
        if self.fromNode not in Nodes:
            Nodes.append(self.fromNode)
        if self.toNode not in Nodes:
            Nodes.append(self.toNode)
    
    def updateDCMat(self, M, B):
        '''Updates the M & B matrices according to the Curent source'''
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
            B[fromNodeInd] -= self.compValue
        except AssertionError:
            pass
        try:
            assert self.toNode != 'GND'
            B[toNodeInd] += self.compValue
        except AssertionError:
            pass
        B[toNodeInd] += self.compValue
        B[fromNodeInd] -= self.compValue
    
    def updateACMat(self, M, B):
        '''Updates the M & B matrices according to the Curent source'''
        fromNodeInd = Nodes.index(self.fromNode)
        toNodeInd = Nodes.index(self.toNode)
        try:
            assert self.fromNode != 'GND'
            B[fromNodeInd] -= self.compValue/2 + 0j
        except AssertionError:
            pass
        try:
            assert self.toNode != 'GND'
            B[toNodeInd] += self.compValue/2 + 0j
        except AssertionError:
            pass
        B[fromNodeInd] -= self.compValue/2 + 0j
        B[toNodeInd] += self.compValue/2 + 0j

# Parsing the file to create classes
try:
    assert len(sys.argv) == 2
except AssertionError:
    print("ERROR: Please give valid no.of arguments")
    sys.exit()

filename = sys.argv[1].strip()

try:
    file = open(filename, 'r')
    isAC = checkAC(file)
    file.close()
    ckt, acfreqs = parseCkt(filename)
    file.close()
    comps = parseComp(ckt)
except FileNotFoundError:
    print("ERROR: Please give a valid file path")

# Creating element objects accordingly

for comp in comps:
    identifier = comp[0][0]
    if identifier == "R":
        # The element is Resistor
        try:
            assert len(comp) == 4
        except AssertionError:
            print(f"ERROR: Incorrect definition of Resistor {comp[0]}")
            sys.exit()
        circuit.append(Resistor(comp))

    elif identifier == "L":
        # The element is Inductor
        try:
            assert len(comp) == 4
        except AssertionError:
            print(f"ERROR: Incorrect definition of Inductor {comp[0]}")
            sys.exit()
        circuit.append(Inductor(comp))

    elif identifier == "C":
        # The element is Capacitor
        try:
            assert len(comp) == 4
        except AssertionError:
            print(f"Incorrect definition of Capacitor {comp[0]}")
            sys.exit()
        circuit.append(Capacitor(comp))

    elif identifier == "V":
        # The element is Voltage Source
        # Now, we have to give distinction of being it as an AC source or DC source
        if isAC:
            # The source is probably AC
            # Format is V... n1 n2 ac Vp-p phase
            for line in acfreqs:
                List = line.split()
                if comp[0] == List[1]:
                    Phase = comp.pop(-1)
                    Type = comp.pop(3)
                    # print(len(comp))
                    # sys.exit()
                    try:
                        assert len(comp) == 4
                    except AssertionError:
                        print(f"ERROR: Incorrect definition of Voltage Source {comp[0]}")
                        sys.exit()
                    circuit.append(VoltageSource(comp))
                    circuit[-1].phase = float(Phase)
                    circuit[-1].type = Type
                    circuit[-1].frequency = float(List[-1])
                    w = float(List[-1])*2*np.pi
                else:
                    # The source is DC in AC circuit
                    # Format is V... n1 n2 dc value
                    Type = comp.pop(3)
                    try:
                        assert len(comp) == 4
                    except AssertionError:
                        print(f"ERROR: Incorrect definition of Voltage Source {comp[0]}")
                        sys.exit()
                    circuit.append(VoltageSource(comp))
                    circuit[-1].type = Type
        else:
            # Format is V... n1 n2 Value
            try:
                assert len(comp) == 4
            except AssertionError:
                print(f"ERROR: Incorrect definition of Voltage Source {comp[0]}")
                sys.exit()
            circuit.append(VoltageSource(comp))
        VoltageSources.append(comp[0])
    
    elif identifier == "I":
        # The element is Current Source
        # Now, we have to give distinction of being it as an AC source or DC source
        if isAC:
            # The source is probably AC
            # Format is I... n1 n2 ac Ip-p phase
            for line in acfreqs:
                List = line.split()
                if comp[0] == List[1]:
                    Phase = comp.pop(-1)
                    Type = comp.pop(3)
                    try:
                        assert len(comp) == 4
                    except AssertionError:
                        print(f"ERROR: Incorrect definition of Current Source {comp[0]}")
                        sys.exit()
                    circuit.append(CurrentSource(comp))
                    circuit[-1].phase = float(Phase)
                    circuit[-1].type = Type
                    circuit[-1].frequency = float(List[-1])
                    w = float(List[-1])*2*np.pi
                else:
                    # The source is DC in AC circuit
                    # Format is I... n1 n2 dc value
                    Type = comp.pop(3)
                    try:
                        assert len(comp) == 4
                    except AssertionError:
                        print(f"ERROR: Incorrect definition of Voltage Source {comp[0]}")
                        sys.exit()
                    circuit.append(CurrentSource(comp))
                    circuit[-1].type = Type
        else:
            # Format is I... n1 n2 Value
            try:
                assert len(comp) == 4
            except AssertionError:
                print(f"ERROR: Incorrect definition of Current Source {comp[0]}")
                sys.exit()
            circuit.append(CurrentSource(comp))

# Check if component names are repeating in the netlist file
compNames = [i.compName for i in circuit]
try:
    assert len(compNames) == len(set(compNames))
except AssertionError:
    print('ERROR: There are repeating component names in the circuit')
    sys.exit()

# Now we have to perform MNA(Modified Nodal Analysis) on the circuit we got
if isAC:
    solveAC()
else:
    solveDC()

# END OF PROGRAM