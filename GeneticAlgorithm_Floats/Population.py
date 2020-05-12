import math
import numpy as np
from numpy.random import randint
import random
import operator
from ast import literal_eval
import struct
import statistics 

class Population():
    def __init__(self, limitA, limitB, gens):
        self.limitA = limitA
        self.limitB = limitB
        self.gens = gens

    #Function Holder Table
    def functionHolderTable(self, x1, x2):
        value = pow(x1,2) + pow(x2,2)
        return value

    #Creating individuals for Initial population
    def createIndividual(self, genes_number, limitA, limitB):
        individual = [round(random.uniform(limitA, limitB),1) for x in range(genes_number)]
        return individual 

    #Initial population
    def createPopulation(self, individuals, genes_number, limitA, limitB):
        return [self.createIndividual(genes_number, limitA, limitB) for x in range(individuals)]

    #Run choosen function
    def fitness_for_function(self, population):
        result = [[(self.functionHolderTable(individual[0], individual[1])), individual] for individual in population]
        return sorted(result, key=operator.itemgetter(0), reverse=False)

    #SELECTION
    #The best individuals
    def selectionBestFitness(self, fitted_population, procentage, minimum="true"):
        individuals_number = int(len(fitted_population) * (procentage / 100))
        #find minimum
        if minimum == "true":
            return [x for x in sorted(fitted_population, key=operator.itemgetter(0))[0:individuals_number]]
        #find maximum
        elif minimum == "false":
            return [x for x in sorted(fitted_population, key=operator.itemgetter(0), reverse=True)[0:individuals_number]]

    #Competition
    def selectionCompetition(self, fitted_population, procentage, minimum="true"):
        winners = []
        participants = []
        population = fitted_population.copy()
        desiredPopulation = int(len(fitted_population) * (procentage / 100))
        participantsInGroup = 4
        while(len(population) > desiredPopulation):
            while(len(population) > 0):
                for i in range(participantsInGroup):
                    if(len(population) > 0):
                        random.shuffle(population)
                        participants.append(population.pop())
                    else:
                        break
                if minimum == "true":
                    winners.append(sorted(participants, key=operator.itemgetter(0))[0])
                else:
                    winners.append(sorted(participants, key=operator.itemgetter(0))[0], reverse=True)
                participants = [] 
            population = winners
            winners = []
        return population

    #Rank
    def selectionRank(self, fitted_population, minimum="true"):
        if(minimum == "true"):
            fitted_population.sort(key=operator.itemgetter(0))
        else:
            fitted_population.sort(key=operator.itemgetter(0))
            fitted_population.reverse()
        return [fitted_population[i] for i in range(len(fitted_population)) for j in range(i)]

    #Roulette
    def selectionRoulette(self, fitted_population, desired_size, minimum="true"):
        individualFitness = fitted_population.copy()
        totalSum = 0
        for individual in individualFitness:
            totalSum += individual[0]
        if(minimum == "true"):
            relative_fitness = [(1 / individual[0]) / totalSum if individual[0] != 0 else 0 for individual in individualFitness]
        elif(minimum == "false"):
            relative_fitness = [individual[0] / totalSum for individual in individualFitness]
        probabilities = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]

        desiredPopulation = []
        for n in range(desired_size):
            r = random.random()
            for (i, individual) in enumerate(individualFitness):
                if  r <= probabilities[i]:
                    desiredPopulation.append(individual)
                    break
        return desiredPopulation

    def arithmetic_corssover(self, firstIndividual, secondIndividual, cross_prob):
        if(random.randint(0,100) <= cross_prob):
            k = random.uniform(0,1)
            firstIndividual[0] = k * firstIndividual[0] + (1 - k) * secondIndividual[0]
            firstIndividual[1] = k * firstIndividual[1] + (1 - k) * secondIndividual[1]
            secondIndividual[0] = (1 - k) * firstIndividual[0] + k * secondIndividual[0]
            secondIndividual[1] = (1 - k) * firstIndividual[1] + k * secondIndividual[1]
        return [firstIndividual, secondIndividual]

    def heur_crossover(self, firstIndividual, secondIndividual, cross_prob):
        if(random.randint(0,100) <= cross_prob):
            k = random.uniform(0,1)
            firstIndividual[0] = k * (secondIndividual[0] - firstIndividual[0]) + secondIndividual[0]
            firstIndividual[1] = k * (secondIndividual[1] - firstIndividual[1]) + secondIndividual[1]
        return firstIndividual

    def uniform_mutation(self, individual, point):
        k = random.randrange(0,1,1)
        if k == 0:
            individual[0] = random.uniform(self.limitA,self.limitB)
        else:
            individual[1] = random.uniform(self.limitA,self.limitB)
        return individual

    def change_index_mutation(self, individual, mut_prob):
        if(random.randint(0,100) <= mut_prob):
            individual[0], individual[1] = individual[1], individual[0] 
        return individual

    
    def elitary_method(self, fitted_population, procentage, minimum="true"):
        individuals_number = int(len(fitted_population) * (procentage / 100))
        if(minimum == "true"):
            return [x[1] for x in sorted(fitted_population, key=operator.itemgetter(0))[0:individuals_number]]
        elif(minimum == "false"):
            return [x[1] for x in sorted(fitted_population, key=operator.itemgetter(0), reverse = True)[0:individuals_number]]

    #Selection methon
    def selectiong_chromosomes(self, method, population, procentage, minimum="true"):
        if(method == "selectionBestFitness"):
            return self.selectionBestFitness(population, procentage, minimum)
        elif(method == "selectionCompetition"):
            return self.selectionCompetition(population, procentage, minimum)
        elif(method == "selectionRoulette"):
            return self.selectionRoulette(population, procentage, minimum)


    #Crossover all parents
    def crossing_chromosomes(self, chromosomes_population, crossing_method, cross_prob):
        crossed = []
        for i in range(0,len(chromosomes_population) - 1,2):
            if(crossing_method == "arithmetic_corssover"):
                r = self.arithmetic_corssover(chromosomes_population[i], chromosomes_population[i + 1], cross_prob)
                for bin in r:
                    crossed.append(bin[:])
            elif(crossing_method == "heur_crossover"):
                r = self.heur_crossover(chromosomes_population[i], chromosomes_population[i + 1], cross_prob)
                crossed.append(r[:])
        return crossed[:]

    #Mutate all chromosomes
    def chromosomes_mutation(self, chromosomes, mutation_method, mut_prob):
        mutated_list = []
        for chromosome in chromosomes:
            if(mutation_method == "uniform_mutation"):
                x = self.uniform_mutation(chromosome, mut_prob)
                mutated_list.append(x[:])
            elif(mutation_method == "change_index_mutation"):
                x = self.change_index_mutation(chromosome, mut_prob)
                mutated_list.append(x[:])
        return mutated_list[:]

    def geneticAlgorithm(self, num_generations, num_individuals, proc_survived, minimum, selection, crossing, mutation, cross_prob, mut_prob, elitary, elit_prob):
        population = self.createPopulation(num_individuals,self.gens,self.limitA,self.limitB)
        fitted_population = self.fitness_for_function(population)
        iterations_results = []
        iterations_results.append(fitted_population)

        for generation in range(num_generations):

            selection_result = self.selectiong_chromosomes(selection, fitted_population, proc_survived, minimum)
            changed_chrom = [x[1] for x in selection_result]
            changed_chrom = self.crossing_chromosomes(changed_chrom, crossing, cross_prob)
            changed_chrom = self.chromosomes_mutation(changed_chrom, mutation, mut_prob)

            offspring = changed_chrom
            if(elitary == "yes"):
                elitary_chromosomes = self.elitary_method(fitted_population, elit_prob, minimum)[0:elit_prob]
                offspring = offspring + elitary_chromosomes

            new_Population = offspring
            for individual in selection_result:
                new_Population.append(individual[1])

            for individual in fitted_population:
                if(len(new_Population) < num_individuals):
                    new_Population.append(individual[1])
                else:
                    break
            
            new_Population = new_Population[0:num_individuals]
            fitted_population = self.fitness_for_function(new_Population)
            iterations_results.append(sorted(fitted_population, key=operator.itemgetter(0), reverse=False))

        return iterations_results

