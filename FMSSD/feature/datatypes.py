"""This file contains the classes of the data types used.
   Inspired by ccheminfolib
"""
import sys
import math
import numpy as np
from copy import deepcopy

##datatypes
BASE		=999
ATOM 		= 0
MOLECULE 	= 2
DESCRIPTOR 	= 3
GRIDPOINT 	= 4
GRID 		= 5
POINT		= 6
##Atomic properties calculated by DFT
MULLIKEN_CHARGE = 0
NAO_CHARGE = 1
NMR_SHIELDING = 2
# Properties obtained using RDKit
TOTAL_VALENCE = 3
AROMATIC = 4
HYBRIDIZATION = 5
DEGREE = 6
FORMAL_CHARGE = 7

##Error types
FAIL = 100
SUCCESS = 101
##some other miscellaneous stuff
NO_OBSERVABLE = -999
##Datatype base class
class Datatype(object):
	"""Abstract class for datatypes.
	
	Subclasses defined by cchemlib:
		Atom, Molecule, Descriptor, Gridpoint, Grid
	"""
	
	def __init__(self, type = BASE):
		self.datatype = type

##Point Datatype
class Point(Datatype):
	"""A simple point in 3 dimensional space.
	
	Used to easily work with distances and coordinates.
	"""
	
	
	def __init__(self, ID, label, x, y, z):
		#copy some infos
		self.ID = ID
		self.label = label
		self.x = np.float64(x)
		self.y = np.float64(y)
		self.z = np.float64(z)
		#call the superclass __init__
		super(Point, self).__init__(POINT)
	#get the distance from a second point
	
	def get_distance_to_point(self, other):
		if isinstance(other, Point):
			return math.sqrt((self.x-other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
		else:
			return NotImplemented
			
			
	#wrap the distance calculation in subtraction operation
	def __sub__(self, other):
	
		return self.get_distance_to_point(other)
		
		
	def __eq__(self, other):
	
		if isinstance(other, Point):
			if self.get_distance_to_point(other) < 0.001:
				return True
			else:
				return False
		else:
			return NotImplemented
			
##Atom Datatype
class Atom(Datatype):
	"""Atom datatype. 
	
	Molecules are made up of atoms
	"""
	
	
	def __init__(self, ID, label):
		#identification
		self.ID = ID       					#atom number in molecule
		self.label = label 					#label in original molecule -- generally element + ID#
		self.coord = None					#Point datatype
		self.layer_id = -1					#ID of grid layer
		self.type = None				#from atomtypes.py -- Tripos definitions
		self.mulliken_charge = 0
		self.NAO_charge = 0
		self.NMR_Shielding = 0	
		# Properties obtained using RDKit
		self.explicit_valence = 0
		self.implicit_valence = 0
		self.total_valence = 0
		self.aromatic = 0
		self.hybridization = 0
		self.degree = 0
		self.total_degree = 0
		self.formal_charge = 0
		super(Atom, self).__init__(ATOM)
	##get the distance from a second atom
	
	def get_distance_from_atom(self, other):
		if isinstance(other, Atom):
			return self.coord-other.coord
		else:
			return NotImplemented
				
	def __sub__(self, other):
		if isinstance(other,Atom):
			return self.get_distance_from_atom(other)

##Descriptor Datatype
class Descriptor(Datatype):
	"""Descriptor datatype.

	"""

	
	def __init__(self, type, value):
	
		self.type = type
		self.value = np.float64(value)
		super(Descriptor, self).__init__(DESCRIPTOR)
		
		
##Gridpoint Datatype
class Gridpoint(Datatype):
	"""Gridpoint datatype.
	"""
	
	def __init__(self, ID, point, descriptors = {}):
	
		self.ID = ID
		self.coord = point
		self.descriptors = descriptors
		self.empty_flag = True#It is used to delete empty grid points.
		self.esp_flag = False
		self.vdw_flag = False
		super(Gridpoint, self).__init__(GRIDPOINT)
		
		
	##add/modify descriptors
	def add_descriptor(self, type, descriptor): 
	
		self.descriptors[type] = deepcopy(descriptor)
		
		
	def get_distance_from_gridpoint(self, other):
		if isinstance(other, Gridpoint):
			return self.coord-other.coord
		else:
			return NotImplemented
	def get_distance_from_point(self, other):
		if isinstance(other, Point):
			return self.coord-other
		else:
			return NotImplemented
	def __sub__(self, other):
		if isinstance(other, Gridpoint):
			return self.get_distance_from_gridpoint(other)
		elif isinstance(other, Point):
			return self.get_distance_from_point(other)
		else:
			return NotImplemented
	def __eq__(self, other):
		if isinstance(other, Gridpoint):
			if self.get_distance_from_gridpoint(other) < 0.001:
				return True
			else:
				return False
		else:
			return NotImplemented

##Molecule Datatype
class Molecule(Datatype):
	"""Molecule datatype.
	"""
	
	def __init__(self,  label):
		#self.ID = ID
		self.label = label
		self.atoms = {}						#atoms 
		self.bonds = {}						#bonds
		self.grid = {}
		self.grind = {}
		self.grind_points = {}
		self.grid_populated = False
		self.descriptors_calculated = False
		self.n_atoms = 0
		self.n_bonds = 0
		# Molecular level properties
		self.homo_energy = 0
		self.lumo_energy = 0
		self.dipole = 0
		self.volume = 0
		self.single_energy = 0
		
		self.spacing = 0
		self.weight = 1.0
		super(Molecule, self).__init__(MOLECULE)
	
# 	def set_weight(self, weight):
# 		self.weight = weight
	def set_grid(self,grid):
		"""A grid shared by all molecules"""
		self.grid = deepcopy(grid)
	def set_spacing(self,spacing):
		"""
		It's for constructor
		"""
		self.spacing = spacing
		
	def atom_within_gridcell(self,gridpoint, atom):
		"""
		Determine whether the atom is in the grid cell where the grid point is located.
		"""
		if (gridpoint.coord.x-self.spacing/2 < atom.coord.x < gridpoint.coord.x+self.spacing/2) \
			and (gridpoint.coord.y-self.spacing/2 < atom.coord.y < gridpoint.coord.y+self.spacing/2) \
			and (gridpoint.coord.z-self.spacing/2 < atom.coord.z < gridpoint.coord.z+self.spacing/2):
			return True
		else:
			return False
	def calculate_gridpoint_descriptors(self):
		"""Accumulate the atomic properties of the molecule in the grid cell."""
		for atom_id in self.atoms:			
			layer_id = self.atoms[atom_id].layer_id
			print('layer_id: ', layer_id)
			least_dist = 100000
			nearest_gridpoint_id = 100000
			for gridpoint_id in range(len(self.grid[layer_id])):
				dist = self.atoms[atom_id].coord - self.grid[layer_id][gridpoint_id].coord
				if dist < least_dist:
					least_dist = dist
					nearest_gridpoint_id = gridpoint_id
			#self.grid[layer_id][nearest_gridpoint_id].empty_flag = False
			#print('layer_id,nearest_gridpoint_id： ',layer_id,nearest_gridpoint_id)
			#
			steric_hindrance_weight = (layer_id+1)/len(self.grid)
			gridpoint_descriptors = self.get_gridpoint_descriptors(layer_id,nearest_gridpoint_id)
			if len(gridpoint_descriptors) == 0:
				gridpoint_descriptors[MULLIKEN_CHARGE] = str(float(self.atoms[atom_id].mulliken_charge.strip())*steric_hindrance_weight)#设置网格点中的描述符
				gridpoint_descriptors[NAO_CHARGE] = str(float(self.atoms[atom_id].NAO_charge.strip())*steric_hindrance_weight)
				gridpoint_descriptors[NMR_SHIELDING] = str(float(self.atoms[atom_id].NMR_Shielding.strip())*steric_hindrance_weight)
				# RDKit properties
				gridpoint_descriptors[TOTAL_VALENCE] = str(float(self.atoms[atom_id].total_valence)*steric_hindrance_weight)
				gridpoint_descriptors[AROMATIC] = str(float(self.atoms[atom_id].aromatic)*steric_hindrance_weight)
				gridpoint_descriptors[HYBRIDIZATION] = str(float(self.atoms[atom_id].hybridization)*steric_hindrance_weight)
				gridpoint_descriptors[DEGREE] = str(float(self.atoms[atom_id].degree)*steric_hindrance_weight)
				gridpoint_descriptors[FORMAL_CHARGE] = str(float(self.atoms[atom_id].formal_charge)*steric_hindrance_weight)

			else:
				gridpoint_descriptors[MULLIKEN_CHARGE] = str(float(gridpoint_descriptors[MULLIKEN_CHARGE])+\
															 float(self.atoms[atom_id].mulliken_charge.strip())*steric_hindrance_weight)
				gridpoint_descriptors[NAO_CHARGE] = str(float(gridpoint_descriptors[NAO_CHARGE])+\
														float(self.atoms[atom_id].NAO_charge.strip())*steric_hindrance_weight)
				gridpoint_descriptors[NMR_SHIELDING] = str(float(gridpoint_descriptors[NMR_SHIELDING])+\
														   float(self.atoms[atom_id].NMR_Shielding.strip())*steric_hindrance_weight)
				# RDKit properties
				gridpoint_descriptors[TOTAL_VALENCE] = str(float(gridpoint_descriptors[TOTAL_VALENCE])+\
														   float(self.atoms[atom_id].total_valence)*steric_hindrance_weight)
				gridpoint_descriptors[AROMATIC] = str(float(gridpoint_descriptors[AROMATIC])+\
													  float(self.atoms[atom_id].aromatic)*steric_hindrance_weight)
				gridpoint_descriptors[HYBRIDIZATION] = str(float(gridpoint_descriptors[HYBRIDIZATION])+\
														   float(self.atoms[atom_id].hybridization)*steric_hindrance_weight)
				gridpoint_descriptors[DEGREE] = str(float(gridpoint_descriptors[DEGREE])+float(self.atoms[atom_id].degree)*steric_hindrance_weight)
				gridpoint_descriptors[FORMAL_CHARGE] = str(float(gridpoint_descriptors[FORMAL_CHARGE])+\
														   float(self.atoms[atom_id].formal_charge)*steric_hindrance_weight)

	def add_atom(self, atom):
		"""Add atom to the atom dictionary at its location by ID"""
		self.n_atoms = len(self.atoms)
		ID = self.n_atoms + 1
		if atom.ID != ID:
			atom.ID = ID
		self.atoms[atom.ID] = atom
		self.n_atoms = len(self.atoms)
		return SUCCESS
	##get a specific gridpoints descriptors
	def get_gridpoint_descriptors(self, layer_id, gridpoint_id):
		try:
			return self.grid[layer_id][gridpoint_id].descriptors
		except KeyError:
			print("Gridpoint: " + str(gridpoint_id) + " not found!")
			return FAIL
	
	def get_result_descriptors(self):
		"""The descriptors of all grid points in the molecule are combined."""
		#IDS = deepcopy(sorted(self.grid.keys()))
		mulliken_charges = ''
		nao_charges = ''
		nmr_shieldings = ''
		# RDKit properties
		total_valences = ''
		aromatics = ''
		hybridizations = ''
		degrees = ''
		formal_charges = ''
		all_features_of_molecule = ''
		for layer_id in range(len(self.grid)):
			for grid_id in range(len(self.grid[layer_id])):
			
				if len(self.grid[layer_id][grid_id].descriptors) == 0:
					mulliken_charge = '0.'
					nao_charge = '0.'
					nmr_shielding = '0.'

					total_valence = '0.'
					aromatic = '0.'
					hybridization = '0.'
					degree = '0.'
					formal_charge = '0.'					
				else:
					mulliken_charge = self.grid[layer_id][grid_id].descriptors[MULLIKEN_CHARGE].strip()
					nao_charge = self.grid[layer_id][grid_id].descriptors[NAO_CHARGE].strip()
					nmr_shielding = self.grid[layer_id][grid_id].descriptors[NMR_SHIELDING].strip()

					total_valence = self.grid[layer_id][grid_id].descriptors[TOTAL_VALENCE].strip()
					aromatic = self.grid[layer_id][grid_id].descriptors[AROMATIC].strip()
					hybridization = self.grid[layer_id][grid_id].descriptors[HYBRIDIZATION].strip()
					degree = self.grid[layer_id][grid_id].descriptors[DEGREE].strip()
					formal_charge = self.grid[layer_id][grid_id].descriptors[FORMAL_CHARGE].strip()
					
					
				mulliken_charges += (mulliken_charge+',')
				nao_charges += (nao_charge+',')
				nmr_shieldings += (nmr_shielding+',')

				total_valences += (total_valence+',')
				aromatics += (aromatic+',')
				hybridizations += (hybridization+',')
				degrees += (degree+',')
				formal_charges += (formal_charge+',')
				all_features_of_molecule += (mulliken_charge+','+nao_charge+','
											 +nmr_shielding+','+total_valence+','+aromatic+','
											 +hybridization+','+degree+','+formal_charge+',')
		mulliken_charges = mulliken_charges[:-1]
		nao_charges = nao_charges[:-1]
		nmr_shieldings = nmr_shieldings[:-1]
		all_features_of_molecule = all_features_of_molecule[:-1]
		return mulliken_charges,nao_charges,nmr_shieldings,all_features_of_molecule

			
