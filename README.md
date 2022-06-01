# Acknowledgement


This code was built upon the source code used in the study:

Damian Dailisan and May Lim (2020). *Crossover transitions in a bus–car mixed-traffic cellular automata model.* Physica A: Statistical Mechanics and its Applications. https://doi.org/10.1016/j.physa.2020.124861

Forever grateful to Mr. Damian Dailisan and Ma'am May Lim for sharing the source code and in guiding me in understanding it. 

The repository may be found here:
https://github.com/temetski/PedestrianCrowding



# About this repository

I uploaded my notebooks and scripts for my BS Physics thesis, *Ride-hailing driver behavior in single-lane and multi-lane traffic*. These notebooks include both the simulation notebooks and the analysis notebooks. I would upload the data but they are too many for now, so I might upload them at a different time.




# Using this repository
```
python setup.py build_ext --inplace 
```

Use this to generate a .cpp file to enable importing the traffic model package. Repeat this when editing the .pyx file.

I included the actual notebooks that I used in generating my final plots. These notebooks are very messy and might be difficult to understand. When I have the time, I'll try to make cleaner notebooks.

I would also like to note that I ran the simulations in our laboratory server, and all in all generating all the *correct* plots would probably total 12+ hours of simulation. I have not tried running them on my local machine, but I doubt I would have gotten faster simulation times.

