"""
Select rows based on some criterions from the ADHD data set.
Will return string of location for IDs
"""

import pandas as pd 	# for reading the csv with suppl data 
import os 				# for creating paths
import re 				# for removing [] in the numpy array
import numpy as np 		# transforming the pandas column to array


class adhd:
	"""
	Parameters
	----------
	csvLocation: String
		location of csv with supplementary information on subjects.
		Path has to be all the way to the file itself.

	niftyLocation: String
		location of data with niftys organized such that each subject
		has a folder named after ID 

	site: int
		which Site to get IDs for

	numbSubjects: int
		number of subjects to return the folder location for
	"""

	def __init__(self, csvLocation = None, niftyLocation = None,
		         site = None, numbSubjects = None):
		self._csvLocation = csvLocation
		self._niftyLocation = niftyLocation
		self._site = site
		self._numbSubjects = numbSubjects

	# --- GETTERS & SETTERS --- #

	@property
	def csvLocation(self):
		return self._csvLocation
	@csvLocation.setter
	def csvLocation(self, strLoc):
		# input validation
		if isinstance(strLoc, String):
			self._csvLocation = strLoc
		else:
			raise TypeError("Must be String")

	@property
	def niftyLocation(self):
		return self._niftyLocation
	@niftyLocation.setter
	def csvLocation(self, strLoc):
		if isinstance(strLoc, String):
			self._niftyLocation = strLoc
		else:
			raise TypeError("Must be String")

	@property
	def site(self):
		return self._site
	@site.setter
	def site(self, siteInt):
		if isinstance(siteInt, int):
			self._site = siteInt
		else:
			raise TypeError("Must be int")

	@property
	def numbSubjects(self):
		return self._numbSubjects
	@numbSubjects.setter
	def numbSubjects(self, numbSubInt):
		if isinstance(numbSubInt, int):
			self._numbSubjects = numbSubInt
		else:
			raise TypeError("Must be int")

	# ---------------------------------- #
	"""
		Should return a list of Strings with the 
		locations of valid repositories in the data
		folder. These should be based on the site
		requested
	"""
	def listOfLocations(self):
		# check that the paths are correctly specified...
		if(self._fileExists(self._niftyLocation)):
			if(self._fileExists(self._csvLocation)):
				# begin by reading the csv
				supplementaryData = pd.read_csv(self._csvLocation)
				# subset to the site (where the data has been gathered)
				# only keep ID column for now
				vec = supplementaryData.loc[(supplementaryData["Site"] == self._site)]
				vec = vec.loc[:, ["ID"]]
				# make it to np.array
				vec = np.array(vec)
				# loop through the vector and check whether the user id 
				# exists in the data repository
				subjectList = []
				# store the stopping conditions for the loop as a dictionary
				stopDict = {'foundNumbOfSubjects': False , 'iter' : 0}
				# loop should stop as soon as any of these conditions has
				# been met
				while(stopDict['foundNumbOfSubjects'] != False or stopDict['iter'] < len(vec)):
					# all the files are of the structure
					# ID_rest_tshift_RPI_voreg_mni.nii
					file = re.sub("[\[\]]", "", np.array_str(vec[stopDict['iter']]))
					if(len(file)) < 7:
						file = self._fileNameUpdater(file)
					tempLoc = os.path.join(self._niftyLocation, file, 
						file + "_rest_tshift_RPI_voreg_mni.nii")
					# print(tempLoc)
					if(self._fileExists(tempLoc)):
						subjectList.append(tempLoc)
						# update dictionary if we have found the specified number of
						# subjects
						if(len(subjectList) == self._numbSubjects):
							stopDict["foundNumbOfSubjects"] = True
					# update iter in dictionary
					stopDict['iter'] += 1
				return subjectList
			else:
				raise FileNotFoundError("Path for csv not found")
		else: 
			raise FileNotFoundError("Path for niftyLocation not found")

	"""
		Helper function to find whether a file excists
	"""
	def _fileExists(self, path):
		try:
			st = os.stat(path)
		except os.error:
			return False
		return True

	"""
		Helper function to makes sure that ID's which are not
		length 7 have 0's added in the beginning until this condition is met
	"""
	def _fileNameUpdater(self, name):
		# how many values are missing for the name to be length 7
		diff = 7 - len(name)
		for i in range(diff):
			# just paste zeroes until its length 7
			name = "0" + name
		return name


