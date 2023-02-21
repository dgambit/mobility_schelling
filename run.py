import pandas as pd
import mesa
import time
import os
from schelling import SchellingAgent, SchellingModel

mobility_params = [{"model" :"classic"}]


for a in range(0,6):
    for b in range(0,11):
        mobility_params.append({"model" :"gravity", "alpha":a/2, "beta":-b/2})
    
    

side = 50
density = 0.7
population = 0.3
homophily = 0.3
town = True

folder = f's{side}_d{density}__p{population}_h{homophily}_twn{town}{hex(time.localtime().tm_mday+time.localtime().tm_min)}_full'
os.mkdir(folder)

params = {"side": [side], "density":[density],"population" : [population], 
          "homophily" : [homophily], "town" : [town],"mobility": mobility_params}

for batch_run in range(10):
	print("batch_run n. ", batch_run)
	iterations = 1
	results = mesa.batch_run(
	    SchellingModel,
	    parameters=params,
    	    iterations=iterations,
	    max_steps=500,
  	  number_processes=1,
 	   data_collection_period=1,
    	display_progress=True)
	

	print("results df")
	results = pd.DataFrame(results)
	t = f'{time.localtime().tm_year}_{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_{time.localtime().tm_min}'
	print("output: ", t)
	results.to_csv(f'{folder}/{t}_{batch_run}.csv')

