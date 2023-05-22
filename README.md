# Mobility constraints in segregation models
## Table of contents
1. [Citing](#citing)
2. [Packages](#packages)
3. [Abstract](#abstract)
4. [Data Availability](#data-availability)
5. [Structure of the repository](#structure-of-the-repository)
6. [Analysis](#analysis)

# Citing
In this repository you can find the code for running our model and to replicate the analysis conducted in our paper.
If you use the code in this repository, please cite our paper:

* Daniele Gambetta, Giovanni Mauro, and Luca Pappalardo. "Mobility constraints in segregation models." 
arXiv preprint arXiv:2305.10170 (2023).*

```
@article{gambetta2023mobility,
  title={Mobility constraints in segregation models},
  author={Gambetta, Daniele and Mauro, Giovanni and Pappalardo, Luca},
  journal={arXiv preprint arXiv:2305.10170},
  year={2023}
}
```
# Packages
For running notebooks and scripts of this project you must install the following Python packages:
```
  mesa
  pandas
  matplotlib
  numpy
  altair
  scikit-mobility
```

# Abstract
Since the development of the original Schelling model of urban segregation, several enhancements have been proposed, but none have considered the impact of mobility constraints on model dynamics. 
Recent studies have shown that human mobility follows specific patterns, such as a preference for short distances and dense locations. This paper proposes a segregation model incorporating mobility constraints to make agents select their location based on distance and location relevance.
Our findings indicate that the mobility-constrained model produces lower segregation levels but takes longer to converge than the original Schelling model. 
We identified a few persistently unhappy agents from the minority group who cause this prolonged convergence time and lower segregation level as they move around the grid centre. 
Our study presents a more realistic representation of how agents move in urban areas and provides a novel and insightful approach to analyzing the impact of mobility constraints on segregation models. 
We highlight the significance of incorporating mobility constraints when policymakers design interventions to address urban segregation.


# Data Availability
The data generatd and analyzed in our work are the ones gnerated by our simulations that you can replicate by running the scripts and notebooks of this repository.


# Structure of the repository
In the **main** level of the repo you can find four folders:

- ```notebooks```: contains Jupyter Notebooks to generate analysis and visualizations of the paper.

- ```results_data```: contains the data generated by our models for the analysis of the main paper and for the analysis of the supplementary notes. 

- ```models```: contains python scripts for MESA implementation of the Model Agent classes of our simulation and pyhon scripts to generate results of simulations.

- ```figures```: contains figures of main paper, generated with notebooks of ```notebooks``` and data in ```results_data```.

<!---

- ```schellingmob.py``` python script containing the MESA implementation of the Model and Agent classes of our simulation
- 
    - 
- ```supplementary.ipynb```
    - Jupyter Notebook to generate analysis and visualizations of supplementary notes
- ```generator_gravity_puagents.py``` 
    - Python code for replicating the analysis of the mobility of the persistently unhappy agents in gravity model.

The folder ```data_main``` contains the data generated by our models for the analysis of the main paper. 
The folder ```data_supplementary``` contains the data generated by our models 

-->

# Analysis

To run our model for replicate the analysis use three jupyter notebooks of this repository.
In particular: 
- Figure 1 is generated by ```fig1.ipynb``` without any supplementary data
- Figure 2 is generated by ```fig2.ipynb``` and data contained in subfolders of data_main
- Figure 3 is generated by ```fig3.ipynb``` and data contained in data_main (```agensts_pu.csv``` and ```suburbia_analysis.csv```)
- Supplementary images are generated by ```supplementary.ipynb``` and data contained in ```results_data/results_data_supplementary```

Regarding data generation process:
- Data in subfolders of ```results_data/results_data_main``` and data of ```results_data/results_data_supplementary``` are generated by ```run.py ```code, using different configuration parameters for each model
- ```agensts_pu.csv``` and ```suburbia_analysis.csv``` are generated by ```generator_gravity_puagents.py```


[figure_1](https://github.com/dgambit/mobility_schelling/blob/main/figure1.pdf)
[figure_2](https://github.com/dgambit/mobility_schelling/blob/main/figure2.pdf)
[figure_3](https://github.com/dgambit/mobility_schelling/blob/main/figure3.pdf)


