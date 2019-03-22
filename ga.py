import numpy as np
from sklearn.metrics import log_loss
from random import randint

class GA:
	def __init__(self, pop_size, num_mating, fitness_func):
		self.pop_size = pop_size
		self.num_mating = num_mating
		self.fitness_func = fitness_func

		self.pop = None

	def build_pop(self, model):
		params = model.get_params()

		self.pop = []

		for i in range(self.pop_size):
			p = []

			for param_dict in params:
				aux = {}

				for key, value in param_dict.items():
					aux[key] = np.random.choice(a=[False, True], size=value.shape)

				p.append(aux)

			self.pop.append(p)

	def fit(self, model, train_x, train_y):
		if self.pop == None:
			self.build_pop(model)

		fitness = self.fitness(model, train_x, train_y)

		parents = self.mating_pool(fitness)

		# offspring_crossover = self.crossover(parents)
		# self.mutation(offspring_crossover)

		# self.pop[0:parents.shape[0], :] = parents
		# self.pop[parents.shape[0]:, :] = offspring_crossover

	def fitness(self, model, train_x, train_y):
		scores = []

		for p in self.pop:
			model.set_params(p)

			pred = model.forward(train_x)

			scores.append(self.fitness_func(train_y, pred))

		return scores

	def mating_pool(self, fitness):
		#parents = np.empty((self.num_mating, self.pop_size[1], self.pop_size[2]))

		# for parent_num in range(self.num_mating):
		# 	min_fitness_idx = np.where(fitness == np.min(fitness))[0][0]

		# 	parents[parent_num, :, :] = self.pop[min_fitness_idx, :, :]

		# 	fitness[min_fitness_idx] = 999999999

		# return parents

	def crossover(self, parents):
		dim_0 = self.pop_size[0] - parents.shape[0]

		offspring = np.empty((dim_0, self.pop_size[1], self.pop_size[2]))

		crossover_point = np.uint32(self.pop_size[1] * 0.5)

		for k in range(dim_0):
			parent1_idx = k % parents.shape[0]
			parent2_idx = (k + 1) % parents.shape[0]

			offspring[k, 0:crossover_point, :] = parents[parent1_idx, 0:crossover_point, :]
			offspring[k, crossover_point:, :] = parents[parent2_idx, crossover_point:, :]

		return offspring

	def mutation(self, offspring_crossover):
		for k in range(offspring_crossover.shape[0]):
			x = randint(0, offspring_crossover.shape[1] - 1)
			y = randint(0, offspring_crossover.shape[2] - 1)

			offspring_crossover[k, x, y] = not offspring_crossover[k, x, y] 