import numpy as np
from random import randint
from scipy.sparse import rand

class GA:
	def __init__(self, pop_size, num_parents, fitness_func, rand_func, mutation_func):
		self.pop_size = pop_size
		self.num_parents = num_parents

		self.fitness_func = fitness_func
		self.rand_func = rand_func
		self.mutation_func = mutation_func

		self.pop = None
		self.best = None

	def build_pop(self, model):
		params = model.get_params()

		self.pop = []

		for i in range(self.pop_size):
			p = []

			for param_dict in params:
				aux = {}

				for key, value in param_dict.items():
					aux[key] = self.rand_func(value.shape)

				p.append(aux)

			self.pop.append(p)

	def fit(self, model, train_x, train_y):
		if self.pop == None:
			self.build_pop(model)

		fitness = self.fitness(model, train_x, train_y)

		parents = self.get_parents(fitness)

		offspring_crossover = self.crossover(parents)
		self.mutation(offspring_crossover)

		self.pop = parents + offspring_crossover

	def fitness(self, model, train_x, train_y):
		scores = []

		for p in self.pop:
			model.set_params(p)

			pred = model.forward(train_x)

			fitness = self.fitness_func(train_y, pred)
				
			scores.append(fitness)

		best_idx = np.argmin(np.array(scores))

		self.best = self.pop[best_idx]
		self.bestFitness = scores[best_idx]

		return scores

	def get_parents(self, fitness):
		parents = []

		for i in range(self.num_parents):
			min_fitness_idx = np.where(fitness == np.min(fitness))[0][0]

			parents.append(self.pop[min_fitness_idx])

			fitness[min_fitness_idx] = 99999

		return parents

	def crossover(self, parents):
		cross_length = self.pop_size - len(parents)

		offspring = []

		for k in range(cross_length):
			p1 = parents[k % len(parents)]
			p2 = parents[(k + 1) % len(parents)]

			child = []

			for i in range(len(p1)):
				p1_param = p1[i]
				p2_param = p2[i]

				chid_param = {}

				for key, value in p1_param.items():
					chid_param[key] = np.empty(value.shape)

					crossover_point = randint(0, value.shape[0] - 1)

					chid_param[key][0:crossover_point, :] = value[0:crossover_point, :]
					chid_param[key][crossover_point:, :] = p2_param[key][crossover_point:, :]
					chid_param[key] = chid_param[key] > 0

				child.append(chid_param)

			offspring.append(child)

		return offspring

	def mutation(self, offspring_crossover):
		for child in offspring_crossover:

			for param_dict in child:	

				for key, value in param_dict.items():
					value = self.mutation_func(value)