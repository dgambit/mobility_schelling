from schellingmob import SchellingAgent, SchellingModel
import math
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

def distance(p, q, type="euclid"):
    if type=="euclid":
        return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
    
side = 50

cells = [(x,y) for x in range(side) for y in range(side)]
mindfc = np.quantile(np.array([distance(p,(25,25)) for p in cells]), 0.1)
maxdfc = np.quantile(np.array([distance(p,(25,25)) for p in cells]), 0.35)

suburb_cells = [c for c in cells if (distance(c,(side/2, side/2))>mindfc and distance(c,(side/2, side/2))<maxdfc)]
other_cells = [c for c in cells if (distance(c,(side/2, side/2))<mindfc or distance(c,(side/2, side/2))>maxdfc)]


dists=[]

for i in range(int(side/2)):
    for k in range(int(side/2)):
        dists.append(distance((0,0), (i,k)))
        
meandfc = sum(dists)/len(dists)

stats_df = pd.DataFrame({ "alpha":[],
                        "beta":[],
                       "iteration" : [],
                       "p_u_sub":[],
                        "p_u_cand":[]})

stats_agents_df = pd.DataFrame({"alpha":[],
                            "beta":[],
                          "AgentID":[],
                         "Step":[],
                         "type":[]})


for iteration in range(100):
    for alfa_beta in [(0,-3), (3,0), (3,-3)]:
        print(iteration, alfa_beta)
        city = SchellingModel(side = side, density=.7,
                                  mobility={"model":"gravity", "alpha":alfa_beta[0], "beta":alfa_beta[1]},
                                  agents_report=True,
                                  town=True)

        city.datacollect()
        while city.running and city.schedule.steps <= 500:
            city.step()


        agents_df = city.datacollector.get_agent_vars_dataframe().reset_index()
        agents_df["distc"]= agents_df.pos.apply(lambda p: distance(p, (side/2, side/2)))

        unhappy_df = agents_df[agents_df["happy"]==False]
        unhappy_df = unhappy_df.groupby("AgentID").agg({'Step': 'count', 'type':"mean"}).reset_index()
        unhappy_df["type"] = unhappy_df["type"].apply(lambda x : "min" if x==1 else "maj")

        unhappy_df["alpha"] = unhappy_df["type"].apply(lambda x: alfa_beta[0])
        unhappy_df["beta"] = unhappy_df["type"].apply(lambda x: alfa_beta[1])
        stats_agents_df = stats_agents_df.append(unhappy_df)
        

        model_df = city.datacollector.get_model_vars_dataframe()
        
        pct_change_censeg = model_df["center_segregation"].pct_change(periods=5)
        step_treshold = pct_change_censeg.lt(0.02).idxmax()
        
    
        outliers = unhappy_df[unhappy_df["Step"]> np.percentile(unhappy_df["Step"], 95)]["AgentID"]
        outliers = set(list(outliers))


        suburb_agents = agents_df[agents_df["Step"]==step_treshold][agents_df["distc"]>mindfc][agents_df["distc"]<maxdfc]["AgentID"]
        suburb_agents = set(list(suburb_agents))
        
        candidate_agents = agents_df[agents_df["Step"]==step_treshold][agents_df["type"]==1][agents_df["happy"]==False][agents_df["distc"]>mindfc][agents_df["distc"]<maxdfc]["AgentID"]
        candidate_agents = set(list(candidate_agents))
        
        
                
        '''
        center_agents  = set(agent_df[agent_df["Step"]==step_treshold][agent_df["type"]==1][agent_df["happy"]==False][agent_df["distc"]<mindfc]["AgentID"])
        periphery_agents = set(agent_df[agent_df["Step"]==step_treshold][agent_df["type"]==1][agent_df["happy"]==False][agent_df["distc"]>maxdfc]["AgentID"])
        '''
   


        new_iter = pd.DataFrame({ "iteration" : [iteration],
                                  "alpha":[alfa_beta[0]],
                                  "beta":[alfa_beta[1]],
                                  "p_u_sub":[len(outliers.intersection(suburb_agents))/len(outliers)],
                                "p_u_cand":[len(outliers.intersection(candidate_agents))/len(outliers)]})

        stats_df = stats_df.append(new_iter)

    
    
    
stats_df.to_csv("stats_df_DRG_0.95_prob100_suburb0.25.csv")
#agents_df.to_csv("agents_DRG_0.95_prob300.csv")