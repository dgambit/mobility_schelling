import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random

import mesa.time
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from matplotlib.colors import ListedColormap

import altair as alt


def distance(p, q, type="euclid"):
    if type=="euclid":
        return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
    elif type=="moore":
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    elif type=="neumann":
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    
def centertown(pos, side):
    center = (side/2,side/2)
    radius= side /10
    return distance(pos,center) < radius+ side/10


class SchellingModel(Model):
    '''
    Model class for the Schelling segregation model.
    '''
    def __init__(self, side=10, density=0.7, population = 0.3, homophily=0.3,  mobility={"model" :"classic"},
                 policy="random",  torus = False, town= False, town_decay=2, agents_report=False, seed=None,
                rel_dist_random =False):
        super().__init__()

        self.side = side
        self.density = density
        
        if seed != None:
            self.random.seed(seed)

        if type(population) != list:
            self.population = [population, 1-population]
        else:
            self.population = population

        self.homophily = homophily
        self.mobility = mobility
        self.policy = policy
        self.schedule = RandomActivation(self)

        self.radius_dict = {((0,0),0):[(0,0)]}
        self.grid = SingleGrid(side, side, torus= torus)

        self.cell_relevance = {(x, y): 1 for x in range(self.side) for y in range(self.side)}

      
        self.total_happy = 0
        self.total_segregation = None
        self.center_segregation = None
        self.periphery_segregation = None
        self.fsi= None

     

        if town:
            if rel_dist_random:
                self.cell_relevance = {(x,y): random.random() for x in range(self.side) for y in range(self.side)}

            else:
                center = (side/2-0.5, side/2-0.5)
                relevance = lambda pos: side/max(distance(pos,center),side/10)**town_decay
                self.cell_relevance = {(x,y): relevance((x,y)) for x in range(self.side) for y in range(self.side)}
           
        
      
        if agents_report:
            self.datacollector = DataCollector(
                {"total_happy": lambda m: m.total_happy/m.schedule.get_agent_count(),
                 "total_segregation": lambda m: m.total_segregation,
                 "fsi": lambda m: m.fsi,
                 "center_segregation": lambda m: m.center_segregation,
                 "periphery_segregation": lambda m: m.periphery_segregation            
                },
            {
                 "type": lambda a: a.type,
                 "pos": lambda a: a.pos,
                 "happy": lambda a: a.happy,
                 "segreg": lambda a: a.segreg,
                "jump": lambda a: a.jump
                 })
            
        else:
            self.datacollector = DataCollector(
                {"total_happy": lambda m: m.total_happy/m.schedule.get_agent_count(),
                 "total_segregation": lambda m: m.total_segregation,
                 "fsi": lambda m: m.fsi,
                 "center_segregation": lambda m: m.center_segregation,
                 "periphery_segregation": lambda m: m.periphery_segregation            
                })
            
        
        
        self.running = True

        agent_id = 1
        
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]

            if self.random.random() < self.density:
                atype = self.random.choices(range(1,len(self.population)+1), self.population)[0]

                agent = SchellingAgent(agent_id, (x, y), self, atype, homophily, mobility= self.mobility, policy= self.policy)
                agent_id += 1
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        for a in self.schedule.agents:
            a.agent_stats()

            
            
        crosslinks = 0
        for a in self.schedule.agents:
            neighbors = self.grid.iter_neighbors(a.pos, moore=True)
            for b in neighbors:
                if b.type != a.type:
                    crosslinks +=1
        crosslinks = crosslinks/2
        
        
        self.fsi_init = crosslinks
        
        self.fsi = crosslinks/self.fsi_init
        
        
        
        self.total_segregation = np.mean([a.segreg for a in self.schedule.agents])
               
        center_town = [a for a in self.schedule.agents if centertown(a.pos, self.side)]
        self.center_segregation = np.mean([a.segreg for a in center_town])
        
        periphery = [a for a in self.schedule.agents if not centertown(a.pos, self.side)]
        self.periphery_segregation = np.mean([a.segreg for a in periphery])
        
        self.total_happy = sum([a.happy for a in self.schedule.agents])
        
        #commentato perche senno crea problemi -> git mesa
        #self.datacollector.collect(self)
        
        
    def datacollect(self):
        self.datacollector.collect(self)

    def step(self):

        self.schedule.step()

        for a in self.schedule.agents:
            a.agent_stats()
            
            
            
            
        crosslinks = 0
        for a in self.schedule.agents:
            neighbors = self.grid.iter_neighbors(a.pos, moore=True)
            for b in neighbors:
                if b.type != a.type:
                    crosslinks +=1
        crosslinks = crosslinks/2
        
        
 
        
        self.fsi = crosslinks/self.fsi_init
        
        
        self.total_segregation = np.mean([a.segreg for a in self.schedule.agents])
        
        center_town = [a for a in self.schedule.agents if centertown(a.pos, self.side)]
        self.center_segregation = np.mean([a.segreg for a in center_town])
        
        periphery = [a for a in self.schedule.agents if not centertown(a.pos, self.side)]
        self.periphery_segregation = np.mean([a.segreg for a in periphery])
        
        self.total_happy = sum([a.happy for a in self.schedule.agents])

        self.datacollector.collect(self)
        
        if self.total_happy == self.schedule.get_agent_count():
            self.running = False


            
    
        
       
    
            
    def show(self,traces=[], labels=False, places= True,  places_val= False, agents=True,unhappy=False, savepdf=None ,figsize=10, ax=None):
        
        if ax is None:
            ax = plt.gca()

        plt.figure(figsize=(figsize, figsize))
        city = [[(agent.type, agent.unique_id) if agent != None else (0,"0") for agent in self.grid[k]] for k in range(self.grid.height)]


        city_id = [[city[i][k][1] for i in range(self.grid.height)] for k in range(self.grid.height)]

        ax.set_xticks(range(self.side))
        ax.set_yticks(range(self.side))

        ax.tick_params(axis='x', colors=(0, 0, 0, 0))
        ax.tick_params(axis='y', colors=(0, 0, 0, 0))

        col_dict = {0:"#ffffff",
                    1: "#ff3300",
                    2: "#3300ff",
                    3: "#00ff33",
                    4: "#ccff00",
                    5:"#cc00ff" }

        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])


        if places:
            city_type = [[self.cell_relevance[(i, k)] for i in range(self.grid.height)] for k in range(self.grid.height)]
        else:
            city_type = [[1 for i in range(self.grid.height)] for k in range(self.grid.height)]
            
        
        img = ax.imshow(city_type, extent=(0, self.side, self.side, 0), cmap="Greys_r")


            
        if places_val:
            for k in range(self.grid.height):
                for i in range(self.grid.width):
                        ax.text(k + 0.5, i +0.5, str(self.cell_relevance[(k,i)])[:3], 
                                     color = "#ff3300", va='center', ha='center', size=6)
                    

        if agents:
            X = [a.pos[0] + 0.5 for a in self.schedule.agents]
            Y = [a.pos[1] + 0.5 for a in self.schedule.agents]
            clist = [col_dict[a.type] for a in self.schedule.agents]
            #plt.scatter(X,Y, s=[15000/(self.side*self.side)]*len(X), marker='o', c=clist, edgecolors="white", linewidth=1)
            ax.scatter(X,Y, s=[15000/(self.side*self.side)]*len(X), marker='s', c=clist, linewidth=1)


        if labels:
            for k in range(self.grid.height):
                for i in range(self.grid.width):
                    if city_id[i][k] != "0":
                        img.text(k + 0.5, i +0.5, city_id[i][k], va='center', ha='center')
                    else:
                        img.text(k + 0.5, i + 0.5, (k,i), va='center', ha='center')

        if unhappy:
            X = [a.pos[0] + 0.5 for a in self.schedule.agents if a.happy==False]
            Y = [a.pos[1] + 0.5 for a in self.schedule.agents if a.happy==False ]
            clist = [col_dict[a.type] for a in self.schedule.agents]
            #plt.scatter(X,Y, s=[15000/(self.side*self.side)]*len(X), marker='o', c=clist, edgecolors="white", linewidth=1)
            ax.scatter(X,Y, s=[15000/(self.side*self.side)]*len(X), marker='x', c="#ffffff", linewidth=2)
            


        #ax.grid(color='w', linewidth= 1/self.side)
        
        for name_agent in traces:
            history = self.datacollector.get_agent_vars_dataframe().reset_index()
            type = history[history["AgentID"] == name_agent]["type"].values[0]
            history_agent = history[history["AgentID"] == name_agent]["pos"]
            history_agent = list(history_agent)
            for kpos in range(1,len(history_agent)):
                o = history_agent[kpos-1]
                d = history_agent[kpos]
                ax.quiver(o[0]+0.5, o[1]+0.5, d[0]-o[0], -d[1]+o[1],color=col_dict[type],
                           units='xy', scale=1, headwidth=5, headlength =7)
        
        
        '''
        if savepdf != None:

            plt.savefig('img/'+savepdf+'.pdf')
            
        '''
        return img
        #plt.show()


class SchellingAgent(Agent):
    def __init__(self, unique_id, pos, model, atype, homophily, mobility , policy):

        super().__init__(unique_id, model)

        #self.hex_id = hex(unique_id)[2:]
        self.pos = pos
        self.jump = 0
        self.type = atype
        self.homophily = homophily
        
        self.mobility = mobility

        self.mobility_model = mobility["model"]

        
        self.policy = policy
        self.next_pos = (0,0)
        self.happy, self.segreg  = None, None


    def agent_stats(self):
        tot_neigh, sim_neigh = 0, 0
        neighbors = self.model.grid.iter_neighbors(self.pos, moore=True)
        for neighbor in neighbors:
            tot_neigh += 1
            if neighbor.type == self.type:
                sim_neigh += 1
        self.segreg = sim_neigh/tot_neigh if tot_neigh!=0 else 0
        self.happy = (tot_neigh==0) or (self.segreg >= self.homophily)
     
    def step(self):
        self.agent_stats()
        if not self.happy:
            self.next_pos = self.next_pos_func()
            self.jump = distance(self.next_pos, self.pos)
            self.model.grid.move_agent(self, self.next_pos)
            
       
    def next_pos_func(self):

        empties = list(self.model.grid.empties)
        empties_good = []

        for e in empties:
            tot_neigh, sim_neigh = 0, 0
            neighbors = self.model.grid.iter_neighbors(e, moore=True)
            for neighbor in neighbors:
                tot_neigh += 1
                if neighbor.type == self.type:
                    sim_neigh += 1
            if (tot_neigh == 0) or (sim_neigh/tot_neigh >= self.homophily):
                empties_good.append(e)

        if self.policy == "good":
            empties = empties_good


        if self.mobility_model == "classic":
            empties_prob = [1/len(empties) for e in empties]
            

        elif self.mobility_model == "gravity":

            alpha = self.mobility["alpha"]
            beta = self.mobility["beta"]

            score_list = [self.model.cell_relevance[e]**alpha * (distance(e, self.pos, "moore")**beta) for e in empties]
            empties_prob = [score / sum(score_list) for score in score_list]


        
        elif self.mobility_model == "radiation":

            score_list = []
            
            val_source = self.model.cell_relevance[self.pos]
            
            
            for e in empties:
                val_dest = self.model.cell_relevance[e]
                radius = distance(self.pos, e, "moore")
                if (self.pos, radius) not in self.model.radius_dict:
                    iter_neigh = self.model.grid.iter_neighborhood(self.pos,
                                                        moore = True,
                                                        include_center= False,
                                                        radius=radius )
                    S = [s for s in iter_neigh]
                    self.model.radius_dict[(self.pos, radius)] = S
                    
                s_r = sum([self.model.cell_relevance[c] for c in self.model.radius_dict[(self.pos, radius)] if self.model.grid.get_cell_list_contents(c) == []])
                #print(self.pos, e, val_source, val_dest,radius, s_r)
                score_list.append(val_dest / ((val_source + s_r) * (val_source + val_dest + s_r)))
            empties_prob = [score / sum(score_list) for score in score_list]
            
            
        elif self.mobility_model == "radiation_inv":

            score_list = []
            val_source = self.model.cell_relevance[self.pos]

            for e in empties:
                val_dest = self.model.cell_relevance[e]
                radius = distance(self.pos, e, "moore")
                if (e, radius) not in self.model.radius_dict:
                    iter_neigh = self.model.grid.iter_neighborhood(e,
                                                        moore = True,
                                                        include_center= False,
                                                        radius=radius )
                    S = [s for s in iter_neigh]
                    self.model.radius_dict[(e, radius)] = S
                    
                s_r = sum([self.model.cell_relevance[c] for c in self.model.radius_dict[(e, radius)] if self.model.grid.get_cell_list_contents(c) == []])
                #print(self.pos, e, val_source, val_dest,radius, s_r)
                score_list.append(val_dest / ((val_source + s_r) * (val_source + val_dest + s_r)))
            empties_prob = [score / sum(score_list) for score in score_list]

            
        index_next_pos =  self.model.random.choices(range(len(empties)), empties_prob)[0]
        return empties[index_next_pos]

