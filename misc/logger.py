import numpy as np
import pickle


class Logger(object):
	def __init__(self, meta_data=dict()):
		self.data = dict()
		self.meta_data = meta_data

	def load(self, file):
		with open(file, "rb") as f:
			x = pickle.load(f)
			self.data = x["data"]
			self.meta_data = x["meta_data"]

	def save(self, file):
		with open(file, "wb") as f:
			x = {"meta_data":self.meta_data, "data": self.data}
			pickle.dump(x, f)

	def store(self, **kwargs):
		for key, value in kwargs.items():
			if key not in self.data.keys():
				self.data[key] = []
			self.data[key].append(value)

	def __str__(self):
		# Print the last values of logger.data
		n = 0
		text = ""
		for key in sorted(self.data.keys()):
			if n > 0:
				text += ", "
			if type(self.data[key][-1]) in [int, float]:
				text += "{}: {:.2e}".format(key, self.data[key][-1])
			else:
				text += "{}: {}".format(key, self.data[key][-1])
			n += 1 
		return text

	def print_all(self):
		print(f" - - - - Step {self.data['step'][-1]} - - - - ")
		for key in list(self.data.keys()):
			print(f"\t{key}:\t{self.data[key][-1]}")
		print('\n')