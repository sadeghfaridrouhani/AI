import numpy
import random

random.seed()

NumDigits = 9

class Population(object):

    def __init__(self):
        self.candidates = []
        return

    def seed(self, noofchromo, given):
        self.candidates = []
        helper = Candidate()
        helper.values = [[[] for j in range(0, NumDigits)] for i in range(0, NumDigits)]
        for row in range(0, NumDigits):
            for column in range(0, NumDigits):
                for value in range(1, 10):
                    if (given.values[row][column] == 0) and not (
                            given.isColumnDuplicate(column, value) or given.isBlockDuplicate(row, column,
                                                                                             value) or given.isRowDuplicate(
                            row, value)):
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        helper.values[row][column].append(given.values[row][column])
                        break

        for p in range(0, noofchromo):
            g = Candidate()
            for i in range(0, NumDigits): 
                row = numpy.zeros(NumDigits, dtype=int)

                for j in range(0, NumDigits): 

                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                while len(list(set(row))) != NumDigits:
                    for j in range(0, NumDigits):
                        if given.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row

            self.candidates.append(g)

        self.updateFitness()
        print("Seeding complete.")
        return

    def updateFitness(self):
        for candidate in self.candidates:
            candidate.updateFitness()
        return

    def sort(self):
        for i in range(len(self.candidates) - 1):
            max = i
            for j in range(i + 1, len(self.candidates)):
                if self.candidates[max].fitness < self.candidates[j].fitness:
                    max = j
            temp = self.candidates[i]
            self.candidates[i] = self.candidates[max]
            self.candidates[max] = temp
        return

class Candidate(object):

    def __init__(self):
        self.values = numpy.zeros((NumDigits, NumDigits), dtype=int)
        self.fitness = None
        return

    def updateFitness(self):

        columnoofchromoount = numpy.zeros(NumDigits, dtype=int)
        blockCount = numpy.zeros(NumDigits, dtype=int)
        coumnSum = 0
        blockSum = 0

        for i in range(0, NumDigits):
            nonzero = 0
            for j in range(0, NumDigits):  
                columnoofchromoount[self.values[j][i] - 1] += 1  

            for k in range(0, NumDigits):
                if columnoofchromoount[k] != 0:
                    nonzero += 1
            nonzero = nonzero / NumDigits
            coumnSum = (coumnSum + nonzero)
            columnoofchromoount = numpy.zeros(NumDigits, dtype=int)
        coumnSum = coumnSum / NumDigits

        for i in range(0, NumDigits, 3):
            for j in range(0, NumDigits, 3):
                blockCount[self.values[i][j] - 1] += 1
                blockCount[self.values[i][j + 1] - 1] += 1
                blockCount[self.values[i][j + 2] - 1] += 1

                blockCount[self.values[i + 1][j] - 1] += 1
                blockCount[self.values[i + 1][j + 1] - 1] += 1
                blockCount[self.values[i + 1][j + 2] - 1] += 1

                blockCount[self.values[i + 2][j] - 1] += 1
                blockCount[self.values[i + 2][j + 1] - 1] += 1
                blockCount[self.values[i + 2][j + 2] - 1] += 1

                nonzero = 0
                for k in range(0, NumDigits):
                    if blockCount[k] != 0:
                        nonzero += 1
                nonzero = nonzero / NumDigits
                blockSum = blockSum + nonzero
                blockCount = numpy.zeros(NumDigits, dtype=int)
        blockSum = blockSum / NumDigits

        if (int(coumnSum) == 1 and int(blockSum) == 1):
            fitness = 1.0
        else:
            fitness = coumnSum * blockSum

        self.fitness = fitness
        return

    def mutate(self, mutationRate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0, 1.1)
        while (r > 1): 
            r = random.uniform(0, 1.1)

        success = False
        if r < mutationRate:  
            while not success:
                row1 = random.randint(0, 8)
                row2 = row1

                fromColumn = random.randint(0, 8)
                toColumn = random.randint(0, 8)
                while fromColumn == toColumn:
                    fromColumn = random.randint(0, 8)
                    toColumn = random.randint(0, 8)

                if given.values[row1][fromColumn] == 0 and given.values[row1][toColumn] == 0:
                    if (not given.isColumnDuplicate(toColumn, self.values[row1][fromColumn])
                            and not given.isColumnDuplicate(fromColumn, self.values[row2][toColumn])
                            and not given.isBlockDuplicate(row2, toColumn, self.values[row1][fromColumn])
                            and not given.isBlockDuplicate(row1, fromColumn, self.values[row2][toColumn])):
                        temp = self.values[row2][toColumn]
                        self.values[row2][toColumn] = self.values[row1][fromColumn]
                        self.values[row1][fromColumn] = temp
                        success = True

        return success

class Given(Candidate):

    def __init__(self, values):
        self.values = values
        return

    def isRowDuplicate(self, row, value):
        for column in range(0, NumDigits):
            if self.values[row][column] == value:
                return True
        return False

    def isColumnDuplicate(self, column, value):
        for row in range(0, NumDigits):
            if self.values[row][column] == value:
                return True
        return False

    def isBlockDuplicate(self, row, column, value):
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
                or (self.values[i][j + 1] == value)
                or (self.values[i][j + 2] == value)
                or (self.values[i + 1][j] == value)
                or (self.values[i + 1][j + 1] == value)
                or (self.values[i + 1][j + 2] == value)
                or (self.values[i + 2][j] == value)
                or (self.values[i + 2][j + 1] == value)
                or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False

class Tournament(object):

    def __init__(self):
        return

    def compete(self, candidates):
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        if f1 > f2:
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.1)
        while r > 1:  
            r = random.uniform(0, 1.1)
        if r < selection_rate:
            return fittest
        else:
            return weakest

class CycleCrossover(object):

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossoverRate):
        child1 = Candidate()
        child2 = Candidate()

        child1.values = numpy.copy(parent1.values)
        child1.fitness = parent1.fitness
        child2.values = numpy.copy(parent2.values)
        child2.fitness = parent2.fitness

        r = random.uniform(0, 1.1)
        while r > 1:  
            r = random.uniform(0, 1.1)

        if r < crossoverRate:
            crossoverPoint1 = random.randint(0, 8)
            crossoverPoint2 = random.randint(1, 9)
            while crossoverPoint1 == crossoverPoint2:
                crossoverPoint1 = random.randint(0, 8)
                crossoverPoint2 = random.randint(1, 9)

            if crossoverPoint1 > crossoverPoint2:
                temp = crossoverPoint1
                crossoverPoint1 = crossoverPoint2
                crossoverPoint2 = temp

            for i in range(crossoverPoint1, crossoverPoint2):
                child1.values[i], child2.values[i] = self.crossoverRows(child1.values[i], child2.values[i])

        return child1, child2

    def crossoverRows(self, row1, row2):
        childRow1 = numpy.zeros(NumDigits)
        childRow2 = numpy.zeros(NumDigits)

        remaining = [i for i in range(1, NumDigits + 1)]
        cycle = 0

        while (0 in childRow1) and (0 in childRow2): 
            if cycle % 2 == 0:
                index = self.findUnused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                childRow1[index] = row1[index]
                childRow2[index] = row2[index]
                next1 = row2[index]

                while next1 != start: 
                    index = self.findValue(row1, next1)
                    childRow1[index] = row1[index]
                    remaining.remove(row1[index])
                    childRow2[index] = row2[index]
                    next1 = row2[index]

                cycle += 1

            else:  
                index = self.findUnused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                childRow1[index] = row2[index]
                childRow2[index] = row1[index]
                next1 = row2[index]

                while next1 != start:  
                    index = self.findValue(row1, next1)
                    childRow1[index] = row2[index]
                    remaining.remove(row1[index])
                    childRow2[index] = row1[index]
                    next1 = row2[index]

                cycle += 1

        return childRow1, childRow2

    def findUnused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if (parent_row[i] in remaining):
                return i

    def findValue(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if (parent_row[i] == value):
                return i

class AI:
    def __init__(self):
        self.given = None
        return

    def load(self, path):
        with open(path, "r") as f:
            values = numpy.loadtxt(f).astype(int)
            self.given = Given(values)
        print("INPUT\n", values)
        return

    def save(self, path, solution):
        with open(path, "w") as f:
            numpy.savetxt(f, solution.values.reshape(NumDigits * NumDigits), fmt='%d')
        return

    def solve(self):
        noofchromo = 100  
        Ne = int(0.6 * noofchromo)
        Ng = 1000  
        Nm = 0  
        staleCount = 0  
        prevFitness = 0

        phi = 0  
        sigma = 1  
        mutationRate = 0.5

        self.population = Population()
        self.population.seed(noofchromo, self.given)

        for generation in range(0, Ng):
            print("Generation %d" % generation)

            bestFitness = 0.0
            bestSolution = self.given
            for c in range(0, noofchromo):
                fitness = self.population.candidates[c].fitness
                if int(fitness) == 1:
                    print("Solution found at generation %d!" % generation)
                    print(self.population.candidates[c].values)
                    return self.population.candidates[c]

                if fitness > bestFitness:
                    bestFitness = fitness
                    bestSolution = self.population.candidates[c].values

            print("Best fitness: %f" % bestFitness)

            nextPopulation = []

            self.population.sort()
            elites = []
            for e in range(0, Ne):
                elite = Candidate()
                elite.values = numpy.copy(self.population.candidates[e].values)
                elites.append(elite)

            for count in range(Ne, noofchromo, 2):
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossoverRate=1.0)

                child1.updateFitness()
                oldFitness = child1.fitness
                success = child1.mutate(mutationRate, self.given)
                child1.updateFitness()
                if success:
                    Nm += 1
                    if child1.fitness > oldFitness: 
                        phi = phi + 1

                child2.updateFitness()
                oldFitness = child2.fitness
                success = child2.mutate(mutationRate, self.given)
                child2.updateFitness()
                if success:
                    Nm += 1
                    if child2.fitness > oldFitness: 
                        phi = phi + 1

                nextPopulation.append(child1)
                nextPopulation.append(child2)

            for e in range(0, Ne):
                nextPopulation.append(elites[e])

            self.population.candidates = nextPopulation
            self.population.updateFitness()

            if Nm == 0:
                phi = 0 
            else:
                phi = phi / Nm

            if phi > 0.2:
                sigma = sigma * 0.998 
            if phi < 0.2:
                sigma = sigma / 0.998 

            mutationRate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
            while mutationRate > 1:
                mutationRate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))

            self.population.sort()

            if generation == 0:
                prevFitness = bestFitness
                staleCount = 1

            elif prevFitness == bestFitness:
                staleCount += 1

            elif prevFitness != bestFitness:
                staleCount = 0
                prevFitness = bestFitness
            if staleCount >= 100:
                print("The population has gone stale. Re-seeding...")
                self.population.seed(noofchromo, self.given)
                staleCount = 0
                sigma = 1
                phi = 0
                mutations = 0
                mutationRate = 0.5

        print("No solution found.", bestSolution)
        return None

s = AI()
s.load("sudoku.json")
solution = s.solve()