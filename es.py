import numpy as np
from random import randint

class ES:
	def __init__(self, pop_size, fitness_func, rand_func):
		self.pop_size = pop_size
		self.fitness_func = fitness_func
		self.rand_func = rand_func

		self.best = None

	def fit(self, model, train_x, train_y):
		if self.best == None:
			self.best = model.get_params()

		pop = self.generate_pop(model)

		fitness = self.fitness(pop, model, train_x, train_y)
		fitness_norm = 1 - (fitness / np.sum(fitness))

		for i in range(self.pop_size):
			p = pop[i]
			f = fitness_norm[i]
			j = 0

			for param_dict in self.best:
				for key, value in param_dict.items():
					value += p[j][key] * f

				j += 1

		self.bestFitness = np.min(fitness)

	def generate_pop(self, model):
		params = model.get_params()

		pop = []

		for i in range(self.pop_size):
			p = []

			for param_dict in params:
				aux = {}

				for key, value in param_dict.items():
					aux[key] = self.rand_func(value.shape)

				p.append(aux)

			pop.append(p)

		return pop

	def merge_params(self, params):
		merge = []

		for param_dict in self.best:
			aux = {}
			other_dict = params[len(merge)]

			for key, value in param_dict.items():
				aux[key] = other_dict[key] + value

			merge.append(aux)

		return merge

	def fitness(self, pop, model, train_x, train_y):
		scores = []

		for p in pop:
			model.set_params(self.merge_params(p))

			pred = model.forward(train_x)

			fitness = self.fitness_func(train_y, pred)
				
			scores.append(fitness)

		return np.array(scores)
