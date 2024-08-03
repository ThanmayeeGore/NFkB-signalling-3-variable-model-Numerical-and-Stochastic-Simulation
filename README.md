# Numerical-and-Stochastic-Simulation of $NF \kappa B$ signalling system


The $NF \kappa B$ signalling system is involved in various cellular functions like inflammation, immune response, apoptosis. The feature that sets apart this system is that it has multiple stimuli or inputs governing the start of the signalling in contrast to others that have fixed or few inputs. So how does the cell recognize the different inputs and and send specific signals of appropriate downstream reactions is a open question. 

The primary genes involved in the signalling are the family of transcription factors $NF \kappa B$ and it's inhibitor $I \kappa B$. Experimental studies have shown that there are oscillations in the nuclear-cytoplasmic shuttling of the $NF \kappa B$ with time period of order of hours. The developed model of coupled nonlinear ordinary differential equations with various levels of complexity explaining the result involves a negative feedback loop constituting the above mentioned genes and saturated degradation causing the time delay. The parametrs of the model control the reaction rates. This system can be implemented in both numerical and stochastic simualtion. The different inputs vary the reaction rates which comes as parameters in the ODEs. Thus changing the parameters in the model gives rise to oscillations with different combinations of features like time period, amplitude, spikeness etc. of oscillations. Consequently we can come up with a phase space of these features. This phase space can then be tessellated so that each polygon represents one state/input. Thus if the oscillations' feature falls into that polygon, it means the input belongs to that particular state. The no of states we get can be inferred to be the number of inputs that can be incorporated (for that particular noise).
