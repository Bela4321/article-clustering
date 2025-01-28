import numpy as np
import random

from pandas.core.computation.expr import intersection

from embeddings.embedding_utils import get_k_highest_values
from embeddings.embedding_utils import clean_and_stem_list

def get_embedding(corpus):
    # turn into a list of lists of words
    corpus = clean_and_stem_list(corpus)
    directions = get_significant_words(corpus)
    embedding = []
    for doc in corpus:
        doc_embedding = np.zeros(len(directions))
        for i, direction in enumerate(directions):
            doc_embedding[i] = doc.count(direction)
        embedding.append(doc_embedding)
    return np.array(embedding), None


def get_significant_words(corpus, num_seed_words=5):
    """Get the most significant words from a corpus."""
    df = calculate_document_frequency(corpus)
    significant_word_applicants = select_significant_word_applicants(corpus, df, num_seed_words)
    genetic_algorithm = GeneticAlgorithm(corpus, df, significant_word_applicants).run()
    return genetic_algorithm.get_significant_words_above_threshold()


def calculate_document_frequency(corpus):
    """Calculate the document frequency (DF) for each term in the corpus."""
    df = {}
    for doc in corpus:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1
    return df



def select_significant_word_applicants(corpus, df, num_seed_words):
    """Select significant words based on document frequency and word pairs."""
    # select the most frequent k words
    frequent_words = get_k_highest_values(df, num_seed_words)
    potential_good_pairs = []
    potential_good_pairs_mu = {}

    for doc in corpus:
        for i, term1 in enumerate(doc[:-1]):
            for term2 in doc[i+1:min(i+4,len(doc)-1)]:  # Window of three words
                if term1 in frequent_words or term2 in frequent_words:
                    potential_good_pairs.append((term1, term2))
                    potential_good_pairs_mu[(term1, term2)] = potential_good_pairs_mu.get((term1, term2), 0) + 1
                    potential_good_pairs_mu[(term2, term1)] = potential_good_pairs_mu.get((term2, term1), 0) + 1
    good_pairs = get_good_pairs(potential_good_pairs, potential_good_pairs_mu, df)
    significant_word_applicants = set()
    for pair in good_pairs:
        significant_word_applicants.add(pair[0])
        significant_word_applicants.add(pair[1])
    return significant_word_applicants



def get_good_pairs(potential_good_pairs, potential_good_pairs_mu, df):
    alpha = 0.5
    good_pairs = []
    for pair in potential_good_pairs:
        mu = potential_good_pairs_mu[pair]
        term1, term2 = pair
        if mu >= min(df[term1], df[term2]) * alpha:
            good_pairs.append(pair)
    return good_pairs


def list_intersection(list1, list2):
    this_intersection = []
    for element in list1:
        if element in list2:
            this_intersection.append(element)
    return this_intersection


class GeneticAlgorithm:
    def __init__(self, corpus, df, significant_words, beta=3, num_generations=25, population_size=600, num_parents=100):
        self.corpus = corpus
        self.df = df
        self.significant_words = significant_words
        self.num_generations = num_generations
        self.beta = beta
        self.population_size = population_size
        self.population = []
        self.num_parents = num_parents
        self.population_hashes = set()
        self.fitness = []
        self.word_occurrences = self.get_word_occurrences()
        self.initialize_population()

    def get_word_occurrences(self):
        # for every word, store the documents in which it occurs
        word_occurrences = {}
        for i, doc in enumerate(self.corpus):
            for term in set(doc):
                if term not in word_occurrences:
                    word_occurrences[term] = []
                word_occurrences[term].append(i)
        return word_occurrences

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = random.sample(self.significant_words, self.beta)
            # make sure that the population is unique
            while hash(tuple(individual)) in self.population_hashes:
                individual = random.sample(self.significant_words, self.beta)
            population.append(individual)
            self.population_hashes.add(hash(tuple(individual)))
        self.population = population

    def calc_fitness(self, individual):
        # fitness = number of documents in which all words of the individual occur
        shared_docs = self.word_occurrences[individual[0]]
        for word in individual[1:]:
            shared_docs = list_intersection(shared_docs,self.word_occurrences[word])
        return len(shared_docs)

    def mutate(self, individual):
        mutated_individual = individual.copy()
        idx = random.randint(0, self.beta-1)
        mutated_individual[idx] = random.choice(list(self.significant_words))
        return mutated_individual

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.beta):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def weighted_choice_pair(self):
        total_fitness = sum(self.fitness)
        if total_fitness == 0:
            weights = [1 / self.population_size for _ in self.fitness]
        else:
            weights = [fitness / total_fitness for fitness in self.fitness]
        choices = random.choices(self.population, weights, k=2)
        return choices[0], choices[1]

    def selection(self):
        self.fitness = [self.calc_fitness(individual) for individual in self.population]
        selection = np.argsort(self.fitness)[-self.num_parents:]
        parents = [self.population[i] for i in selection]
        parent_fitness = [self.fitness[i] for i in selection]
        self.population = parents
        self.fitness = parent_fitness
        self.population_hashes = set([hash(tuple(individual)) for individual in self.population])

    def generation(self):
        self.selection()
        new_population = self.population.copy()
        while len(new_population) < self.population_size:
            parent1, parent2 = self.weighted_choice_pair()
            if random.random() < 0.5:
                child = self.crossover(parent1, parent2)
            else:
                child = self.mutate(parent1)
            if hash(tuple(child)) not in self.population_hashes:
                new_population.append(child)
                self.population_hashes.add(hash(tuple(child)))
        self.population = new_population

    def run(self):
        for _ in range(self.num_generations):
            self.generation()
            print (f"Archieved best fitness: {max(self.fitness)}")
        return self

    def get_significant_words_above_threshold(self, threshold = 2):
        significant_words = set()
        for individual in self.population:
            if self.calc_fitness(individual) >= threshold:
                significant_words.update(individual)
        return significant_words