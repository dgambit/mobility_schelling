from schellingmob import SchellingAgent, SchellingModel
import pandas as pd
import matplotlib.pyplot as plt
import mesa
import numpy as np
import os


#mobility_params = [{"model" :"classic"}]
mobility_params =[]

'''
folder = "distance"
alfa_start, alfa_stop, alfa_step = 0, 0, 1
beta_start, beta_stop, beta_step = -5, 5, 0.1

folder = "relevance"
alfa_start, alfa_stop, alfa_step = 0, 3, 0.1
beta_start, beta_stop, beta_step = 0, 0, 1

folder="gravity"
alfa_start, alfa_stop, alfa_step = 0, 2.5, 0.5
beta_start, beta_stop, beta_step = -5, 0, 1.0
'''

folder = "distance"
alfa_start, alfa_stop, alfa_step = 0, 0, 1
beta_start, beta_stop, beta_step = -5, 5, 0.1

for a in np.arange(alfa_start, alfa_stop+alfa_step, alfa_step):
    for b in np.arange(beta_start, beta_stop+beta_step, beta_step):
        a = round(a,1)
        b = round(b,1)
        mobility_params.append({"model":"gravity", "alpha":a, "beta": b})
        

side = 50
density = 0.7
population = 0.3
homophily = 0.3
town = True


iterations = 10

params = {"side": [side], "density":[density],"population" : [population], 
          "homophily" : [homophily], "town" : [town],"mobility": mobility_params}



superfolder = f'side-{side}_dens-{density}_pop-{population}_hom-{homophily}_town-{town}'

folder = superfolder + "/"+ folder + f'_side-{side}_dens-{density}_pop-{population}_hom-{homophily}_town-{town}'

if not os.path.exists(folder):
    os.makedirs(folder)

for batch in range(10):
    df = mesa.batch_run(SchellingModel,parameters=params,
                        iterations=iterations, max_steps=500, 
                        number_processes=1,data_collection_period=-1, display_progress=True)

    results = pd.DataFrame(df)

    name = f'side-{side}_dens-{density}_pop-{population}_hom-{homophily}_town-{town}_{batch}'


    print("output: ", folder +" " +name)
    results.to_csv(f'{folder}/{name}.csv')
    
