import copy
import numpy as np
from player import Player
from plotting import write_to_file


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def save_generation_information(self, max, min, avg):
        write_to_file(max, min, avg)

    def get_players_fitness(self, players):
        players_fitness = []
        for player in players:
            players_fitness.append(player.fitness)
        return players_fitness

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # (Implement top-k algorithm here)

        sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)

        # save generation information to plotting
        all_fitness_in_generation = self.get_players_fitness(players)
        max = np.max(all_fitness_in_generation)
        min = np.min(all_fitness_in_generation)
        avg = np.mean(all_fitness_in_generation)
        self.save_generation_information(max, min, avg)

        return sorted_players[: num_players]

    def q_tournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)

    def add_gaussian_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)

    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.2

        self.add_gaussian_noise(child.nn.W1, threshold)
        self.add_gaussian_noise(child.nn.W2, threshold)
        self.add_gaussian_noise(child.nn.b1, threshold)
        self.add_gaussian_noise(child.nn.b2, threshold)

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        parents = []
        children = []
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            for player in prev_players:
                parents.append(self.clone_player(player))
        for i in range(0, len(parents), 2):
            child1, child2 = self.reproduction(parents[i], parents[i + 1])
            children.append(child1)
            children.append(child2)
        return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        row_size, column_size = child1_array.shape
        section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:section_1, :] = parent1_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent1_array[section_2:, :]

            child2_array[:section_1, :] = parent2_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent2_array[section_2:, :]
        else:
            child1_array[:section_1, :] = parent2_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent2_array[section_2:, :]

            child2_array[:section_1, :] = parent1_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent1_array[section_2:, :]

    def reproduction(self, parent1, parent2):
        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)

        self.crossover(child1.nn.W1, child2.nn.W1, parent1.nn.W1, parent2.nn.W1)
        self.crossover(child1.nn.W2, child2.nn.W2, parent1.nn.W2, parent2.nn.W2)
        self.crossover(child1.nn.b1, child2.nn.b1, parent1.nn.b1, parent2.nn.b1)
        self.crossover(child1.nn.b2, child2.nn.b2, parent1.nn.b2, parent2.nn.b2)

        self.mutate(child1)
        self.mutate(child2)
        return child1, child2
