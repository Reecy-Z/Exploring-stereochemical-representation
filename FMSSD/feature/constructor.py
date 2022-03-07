"""This file contains the grid constructor for the fibonacci multilayer spherical
   sampling descriptor (FMSSD).
"""
from copy import deepcopy
import math
import datatypes as dt
import numpy as np

class Constructor(object):
	"""superclass for all constructors"""
	def __init__(self):
		print("It's the superclass for all constructors.")
class GridConstructor(Constructor):
	"""creates a grid for a series of molecules"""
	def __init__(self, molecules, spacing=1.0, padding=3.0,
				 homogenize=True):
		# list of molecules
		self.mols = molecules
		# This dictionary will be filled with Gridpoints
		self.grid = {}
		self.reduced_grid = {}
		# spacing between gridpoints
		self.spacing = spacing
		# length to extend grid beyond furthest atomic center from origin
		self.padding = padding
		# a homogenous grid
		self.homogenize = homogenize
		super(GridConstructor, self).__init__()
	def get_origin_of_grid(self):
		"""determines the average center of the aligned molecules"""
		# coordinates
		x = []
		y = []
		z = []
		# append the coordinates of all atoms to the lists
		for mol in self.mols:
			for atom in mol.atoms:
				x.append(mol.atoms[atom].coord.x)
				y.append(mol.atoms[atom].coord.y)
				z.append(mol.atoms[atom].coord.z)
		# set the origin
		self.origin = dt.Point(0,"origin", np.mean(x), np.mean(y), np.mean(z))
		return dt.SUCCESS
	def determine_grid_radius(self):
		"""determines the radius of the outmost layer
		"""
		
		self.radius = 0.0
		# loop through the list of mols to determine the radius
		for mol in self.mols:
			for atom in mol.atoms:
				dist = mol.atoms[atom].coord - self.origin
				if dist > self.radius:
					self.radius = dist
				else:
					pass
		# The maximum radius has been generated,
		# and then padding is added to ensure that
		# each molecule is wrapped by the grid
		self.radius = self.radius + self.padding
	
	def search_grid_by_radius(self, origin, radius, grid_span, num_init, step):
		"""
		Search the grid points with spacing grid_span on a sphere
		Parameters:
			origin - origin coordinates
			radius - radius of spherical surface of this layer
			grid_span - spacing of grid points
			num_init - initial grid points, search from the innermost layer outward
			step - the step size of each grid search
		Returns:
			grid points spaced grid_span apart from each other
		"""    
		if num_init == 1:
			x,y,z = [0.0],[0.0],[0.0]
			#span = 0
			grid = list(zip(x,y,z))
			return np.array(grid),x,y,z
		else:
			phi=(math.sqrt(5)+1)/2-1
			num = num_init
			while True:
				z = [(2*i-1)/num-1 for i in range(1,num+1)]
				z = np.array(z)
				x = [math.sqrt(1-z[i-1]**2)*math.cos(2*math.pi*i*phi) for i in range(1,num+1)]
				x = np.array(x)
				y = [math.sqrt(1-z[i-1]**2)*math.sin(2*math.pi*i*phi) for i in range(1,num+1)]
				y = np.array(y)
				x *= radius
				y *= radius
				z *= radius
				x += origin.x
				y += origin.y
				z += origin.z

				# Spacing between two grid points
				span = math.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2+(z[0]-z[1])**2)        
				#print('span:',span)
				if span < grid_span:
					#Coordinates of grid points on this layer
					grid = list(zip(x,y,z))
					return np.array(grid),x,y,z
				else:
					num += step

	def search_sphere_grid(self, origin, outermost_radius, grid_span, num_init, step):
		"""
	    Generate a multi-layer spherical grid
		Parameters:
			origin - origin coordinates
			outermost_radius - radius of spherical surface of outermost layer
			grid_span - spacing of grid points
			num_init - initial grid points, search from the innermost layer outward
			step - the step size of each grid search
		Returns:
			a multi-layer spherical grid
	    """    
		sphere_grid = []
		xs = []
		ys = []
		zs = []
		#grid_span = 1#
		layer_radius = grid_span
		#Innermost_num_init = 1
		outermost_layers = 1 if outermost_radius%grid_span>0 else 0
		num_layers = int(outermost_radius/grid_span)+outermost_layers
		for i in range(num_layers):
			grid_layer,x,y,z = self.search_grid_by_radius(origin, layer_radius, grid_span, num_init, step)
			sphere_grid.append(grid_layer)
			xs.append(x)
			ys.append(y)
			zs.append(z)
			layer_radius += grid_span
			num_init = len(grid_layer)+1
		
		return sphere_grid,xs,ys,zs

	def generate_initial_grid(self):
		sphere_grid,xs,ys,zs = self.search_sphere_grid(self.origin, self.radius, self.spacing, 1, 10)
		gridpoint_index = 0
		for layer_id, layer_grid in enumerate(sphere_grid):
			self.grid[layer_id] = []
			for grid_id in range(len(layer_grid)):
				coord = dt.Point(gridpoint_index, str(gridpoint_index), layer_grid[grid_id][0], layer_grid[grid_id][1], layer_grid[grid_id][2])
				self.grid[layer_id].append(dt.Gridpoint(gridpoint_index, coord))
				gridpoint_index += 1
			#self.grid[layer_id] = layer_grid
		print('total gridpoint: ', str(gridpoint_index+1))

	def get_atom_layer(self, atom):
		"""
		Get which layer the atom is on.
		Parameters:
			atom - an Atom instance
		"""
		the_radius = atom.coord - self.origin
		layer_id = int(the_radius/self.spacing)
		#atom.layer_id = layer_id
		return layer_id
	
	def set_atom_layer(self, atom):
		"""
		Set which layer the atom is on.
		Parameters:
			atom - an Atom instance
		"""
		the_radius = atom.coord - self.origin
		layer_id = int(the_radius/self.spacing)
		atom.layer_id = layer_id		
		
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

	def molecule_within_gridcell(self, gridpoint, mol):
		"""
		Determine whether the molecule is in the grid cell where the grid point is located.
		"""
		for atom_id in mol.atoms:
			if self.atom_within_gridcell(gridpoint, mol.atoms[atom_id]):
				return True
			else:
				continue
		return False
	
	def get_grid_len(self,grid):
		"""
		Calculate the length of the grid.
		"""
		count = 0
		for layer_id in range(len(grid)):
			count += len(grid[layer_id])
		return count
			
	def reduce_grid(self):
		"""
		Delete unoccupied grid cells.
		"""
		for mol in self.mols:
			#atoms = deepcopy(mol.atoms)
			for atom_id in mol.atoms:			
				layer_id = self.get_atom_layer(mol.atoms[atom_id])
				print('layer_id: ', layer_id)
				# Find the grid point closest to the atom in this layer
				least_dist = 100000
				nearest_gridpoint_id = 100000
				for gridpoint_id in range(len(self.grid[layer_id])):
					#
					dist = mol.atoms[atom_id].coord - self.grid[layer_id][gridpoint_id].coord
					if dist < least_dist:
						least_dist = dist
						nearest_gridpoint_id = gridpoint_id
				self.grid[layer_id][nearest_gridpoint_id].empty_flag = False

		for layer_id in range(len(self.grid)):
			self.reduced_grid[layer_id] = []
			for gridpoint_id in range(len(self.grid[layer_id])):
				if not self.grid[layer_id][gridpoint_id].empty_flag:
					self.reduced_grid[layer_id].append(deepcopy(self.grid[layer_id][gridpoint_id]))					

		print('Reserved grid points: ', self.get_grid_len(self.reduced_grid))
		print("Removed: " + str(self.get_grid_len(self.grid)-self.get_grid_len(self.reduced_grid)) + " gridpoints")
		return dt.SUCCESS

	def generate_grid(self):
			"""
			Generate a multilayer spherical grid shared by molecules.
			"""
			print("Determining origin ...")
			self.get_origin_of_grid()
			print("Origin set at x: " + str(self.origin.x) + " y: " + str(self.origin.y) + " z: " + str(self.origin.z))
			print("Calculating the radius...")
			self.determine_grid_radius()
			print("Grid radius: " + str(self.radius))
			print("Generating an initial multi-layer spherical grid...")
			self.generate_initial_grid()
			print("Number of layers: " + str(len(self.grid)))
			if self.homogenize:
				print("Deleting unoccupied grid cells....")
				self.reduce_grid()
				print("A multi-layer spherical grid has been generated!")
				return deepcopy(self.reduced_grid), len(self.grid)
			else:
				print("GRID generation complete!")
				return self.grid, len(self.grid)
	
	def set_molecules_grid(self, mols):
		"""
		Set the grid of molecules.
		Parameters:
			mols - A list containing Molecule instances
		"""
		for mol in mols:
			print('set_molecule_grid: ', str(mol.label))
			mol.set_grid(self.reduced_grid)
			mol.set_spacing(self.spacing)
			for atom_id in mol.atoms:
				self.set_atom_layer(mol.atoms[atom_id])
