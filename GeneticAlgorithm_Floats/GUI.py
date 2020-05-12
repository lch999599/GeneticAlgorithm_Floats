import wx 
import Population
import time
import datetime
import pandas
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import animation
import operator 
from matplotlib import cm
import statistics

class GeneticAlgorithm(wx.Frame):
    def __init__(self, *args, **kw):
        super(GeneticAlgorithm, self).__init__(*args, **kw)

        panel = wx.Panel(self, wx.ID_ANY)
        
        self.textFunction       = wx.StaticText(panel, wx.ID_ANY, 'FUNCTION: ', (20,20))
        self.textFunction       = wx.StaticText(panel, wx.ID_ANY, 'f(x) =  x^2 + y^2 ', (200,20))

        self.textOptimum        = wx.StaticText(panel, wx.ID_ANY, 'Select Optimalization', (20,80))
        self.comboOptimum       = wx.ComboBox(panel, wx.ID_ANY, 'minimalization', (200,80), choices = ["minimalization", "maximalization"])
        
        self.textGenerations    = wx.StaticText(panel, wx.ID_ANY, 'Number of generations', (20,160))
        self.insertGenerations  = wx.TextCtrl(panel, wx.ID_ANY, '100', (200,160))

        self.textPupulationSize = wx.StaticText(panel, wx.ID_ANY, 'Size of population', (20,200))
        self.insertPupulation   = wx.TextCtrl(panel, wx.ID_ANY, '100', (200,200))

        self.textSelection      = wx.StaticText(panel, wx.ID_ANY, 'Selection method', (20,240))
        self.comboSelection     = wx.ComboBox(panel, wx.ID_ANY, 'Roulette', (200,240), choices = ["Best Fitness", "Competition", "Roulette"])

        self.textSurvived       = wx.StaticText(panel, wx.ID_ANY, 'Number of selected', (20,280))
        self.insertSurvived     = wx.TextCtrl(panel, wx.ID_ANY, '80', (200,280))
        wx.StaticText(panel, wx.ID_ANY, '%', (340,280))

        self.textCrossover      = wx.StaticText(panel, wx.ID_ANY, 'Crossing method', (20,320))
        self.comboCrossover     = wx.ComboBox(panel, wx.ID_ANY, 'Heuristic Crossover', (200,320), choices = ["Arithmetic corssover", "Heuristic Crossover"])
        self.insertCrossover     = wx.TextCtrl(panel, wx.ID_ANY, '50', (400,320))
        wx.StaticText(panel, wx.ID_ANY, '%', (540,320))


        self.textMutation       = wx.StaticText(panel, wx.ID_ANY, 'Mutation method', (20,360))
        self.comboMutation     = wx.ComboBox(panel, wx.ID_ANY, 'Uniform Mutation', (200,360), choices = ["Uniform Mutation","Change index mutation"])
        self.insertMutation     = wx.TextCtrl(panel, wx.ID_ANY, '50', (400,360))
        wx.StaticText(panel, wx.ID_ANY, '%', (540,360))

        self.textElitary        = wx.StaticText(panel, wx.ID_ANY, 'Elitary ', (20,440))
        self.comboElitary       = wx.ComboBox(panel, wx.ID_ANY, 'yes', (200,440), choices = ["yes", "no"])
        self.insertElitary      = wx.TextCtrl(panel, wx.ID_ANY, '2', (400,440))
        wx.StaticText(panel, wx.ID_ANY, '%', (540,440))

        self.button = wx.Button(panel, wx.ID_ANY, 'Start', (20, 480))
        self.button.Bind(wx.EVT_BUTTON, self.onStart)

        self.buttonIterations = wx.Button(panel, wx.ID_ANY, 'Print Iterations', (20, 540))
        self.buttonIterations.Bind(wx.EVT_BUTTON, self.print_iterations)
        self.buttonIterations.Hide()

        self.buttonCSV = wx.Button(panel, wx.ID_ANY, 'Write to .csv', (120, 540))
        self.buttonCSV.Bind(wx.EVT_BUTTON, self.save_to_csv)
        self.buttonCSV.Hide()

        self.buttonPlot = wx.Button(panel, wx.ID_ANY, 'Show Plot', (220, 540))
        self.buttonPlot.Bind(wx.EVT_BUTTON, self.show_plot)
        self.buttonPlot.Hide()

        self.buttonVisual = wx.Button(panel, wx.ID_ANY, 'Show Visualization', (260, 580))
        self.buttonVisual.Bind(wx.EVT_BUTTON, self.show_visual)
        self.buttonVisual.Hide()

        self.textVisual      = wx.StaticText(panel, wx.ID_ANY, 'Set time period', (20,580))
        self.textVisual.Hide()

        self.insertVisual    = wx.TextCtrl(panel, wx.ID_ANY, '0.1', (140,580))
        self.insertVisual.Hide()

        self.buttonBest = wx.Button(panel, wx.ID_ANY, 'Show Best', (320, 540))
        self.buttonBest.Bind(wx.EVT_BUTTON, self.show_best)
        self.buttonBest.Hide()

        self.buttonAverage = wx.Button(panel, wx.ID_ANY, 'Show Average', (420, 540))
        self.buttonAverage.Bind(wx.EVT_BUTTON, self.show_average)
        self.buttonAverage.Hide()

        self.buttonDeviation = wx.Button(panel, wx.ID_ANY, 'Show Standard Deviation', (520, 540))
        self.buttonDeviation.Bind(wx.EVT_BUTTON, self.show_deriveration)
        self.buttonDeviation.Hide()

        self.textElitary        = wx.StaticText(panel, wx.ID_ANY, 'Time ', (20,620))
        self.textResults        = wx.TextCtrl(panel, wx.ID_ANY, ' ', (100,620))
        #self.textResults.Disable()

        self.textYStatic        = wx.StaticText(panel, wx.ID_ANY, 'Y ', (20,660))
        self.textY        = wx.TextCtrl(panel, wx.ID_ANY, ' ', (100,660))
        self.textX1Static        = wx.StaticText(panel, wx.ID_ANY, 'X1 ', (260,660))
        self.textX1        = wx.TextCtrl(panel, wx.ID_ANY, ' ', (340,660))
        self.textX2Static       = wx.StaticText(panel, wx.ID_ANY, 'X2 ', (480,660))
        self.textX2        = wx.TextCtrl(panel, wx.ID_ANY, ' ', (560,660))

    def onStart(self, event):
        global iterations_results
        generations = int(self.insertGenerations.GetValue())
        population_size = int(self.insertPupulation.GetValue())
        survived_size = int(self.insertSurvived.GetValue())

        if(self.comboOptimum.GetValue() == 'minimalization'):
            optimum = "true"
        else:
            optimum = "false"

        if(self.comboSelection.GetValue() == "Best Fitness"):
            selection_method = "selectionBestFitness"
        elif(self.comboSelection.GetValue() == "Competition"):
            selection_method = "selectionBestFitness"
        elif(self.comboSelection.GetValue() == "Roulette"):
            selection_method = "selectionBestFitness"

        if(self.comboCrossover.GetValue() == "Arithmetic corssover"):
            crossing_method = "arithmetic_corssover"
        elif(self.comboCrossover.GetValue() == "Heuristic Crossover"):
            crossing_method = "heur_crossover"

        if(self.comboMutation.GetValue() == "Uniform Mutation"):
            mutation_method = "uniform_mutation"
        elif(self.comboMutation.GetValue() == "Change index mutation"):
            mutation_method = "change_index_mutation"

        cross_pro = int(self.insertCrossover.GetValue())
        mut_pro = int(self.insertMutation.GetValue())
    
        elitary = self.comboElitary.GetValue()
        elit_pro = int(self.insertElitary.GetValue())

        start_time = time.time()

        population = Population.Population(-10,10,2)

        iterations_results = population.geneticAlgorithm(generations, population_size, survived_size, optimum, selection_method, crossing_method, mutation_method, cross_pro, mut_pro, elitary, elit_pro)
        elapsed_time = time.time() - start_time
        self.textResults.SetValue(str(elapsed_time))
        #bestResult = [x for x in sorted(fitted_population, key=operator.itemgetter(0))[0:individuals_number]][0]
        self.textY.SetValue(str(iterations_results[-1][0][0]))
        self.textX1.SetValue(str(iterations_results[-1][0][1][0]))      
        self.textX2.SetValue(str(iterations_results[-1][0][1][1]))  
        self.showButtons()

    def showButtons(self):
        self.buttonIterations.Show()
        self.buttonCSV.Show()
        self.buttonPlot.Show()
        self.buttonBest.Show()
        self.buttonAverage.Show()
        self.buttonDeviation.Show()
        self.textVisual.Show()
        self.insertVisual.Show()
        self.buttonVisual.Show()
        
    def save_to_csv(self, event):
        dt = datetime.datetime.now()
        name = 'result_' + str(dt.day) + '.' + str(dt.month) + '.' + str(dt.year) + '_' + str(dt.hour) + '-' + str(dt.minute) + '-' + str(dt.second) + '.csv'
        n = 0
        with open(name, 'w') as f:
            writer = csv.writer(f)
            for iteration in iterations_results:
                text = "ITERACJA: " + str(n)
                n = n + 1
                writer.writerow([text])
                writer.writerow(['y x1 x2'])
                for object in iteration:
                    values = str(round(object[0],5)) + ", " + str(round(object[1][0],5)) + ", " + str(round(object[1][1],5))
                    writer.writerow([values])
        print("Data saved to CSV file as: ", name)

    def print_iterations(self, event):
        n=0
        data = {}
        for iteration in iterations_results:
            text = "ITERACJA: " + str(n)
            data = {}
            n = n + 1
            print(text)
            for object in iteration:
                data[round(object[0],5)] = [round(object[1][0],5), round(object[1][1],5)]
            for key,val in data.items():
                print("[y]= ", key, " [x1,x2]= ", val)
            print("\n Individuals in population= ", len(data))

    def show_visual(self, event):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plt.ion()
        plt.show()
        for iteration in iterations_results:
            x1=[]
            x2=[]
            y=[]
            for object in iteration:
                x1.append(object[1][0])
                x2.append(object[1][1])
                y.append(object[0])
            ax.clear()
            ax.scatter(x1, x2, y)
            plt.draw()
            plt.pause(float(self.insertVisual.GetValue()))

    def show_plot(self, event):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x1 = [x[1][0] for x in iterations_results[-1]]
        x2 = [x[1][1] for x in iterations_results[-1]]
        y = [x[0] for x in iterations_results[-1]]
        ax.scatter(x1, x2, y)
        plt.show()

    def show_best(self, event):
        plt.xlabel('Iteracja')
        plt.ylabel('Wartość funkcji dla najlepszego')
        plt.title('Zmiana najlepszego osobnika')
        plt.grid(True)
        best = []
        for iteration in iterations_results:
            if self.comboOptimum.GetValue() == 'minimalization':
                x=[x[0] for x in sorted(iteration, key=operator.itemgetter(0))]
            else:
                x=[x[0] for x in sorted(iteration, key=operator.itemgetter(0), reverse=True)]
            best.append(x[0])
        plt.plot(best)
        plt.show()
   
    def show_average(self, event):
        average = []
        for iteration in iterations_results:
            if self.comboOptimum.GetValue() == 'minimalization':
                x=[x[0] for x in sorted(iteration, key=operator.itemgetter(0))]
            else:
                x=[x[0] for x in sorted(iteration, key=operator.itemgetter(0), reverse=True)]
            average.append(statistics.mean(x))
        plt.plot(average)
        plt.xlabel('Iteracja')
        plt.ylabel('Wartość średnia przystosowania')
        plt.title('Zmiana średniej przystosowania')
        plt.grid(True)
        plt.show()

    def show_deriveration(self, event):
        standard_derivation= []
        for iteration in iterations_results:
            if self.comboOptimum.GetValue() == 'minimalization':
                x = [x[0] for x in sorted(iteration, key=operator.itemgetter(0))]
            else:
                x = [x[0] for x in sorted(iteration, key=operator.itemgetter(0), reverse=True)]
            standard_derivation.append(statistics.stdev(x))
        plt.plot(standard_derivation)
        plt.xlabel('Iteracja')
        plt.ylabel('Wartość odchylenia standardowego')
        plt.title('Zmiana odchylenia standardowego')
        plt.grid(True)
        plt.show()
        
if __name__ == '__main__':
    app = wx.App()
    frame = GeneticAlgorithm(None, title='x2 y2 TABLE FUNCTION', size = (750,750))
    frame.Show()
    app.MainLoop()