## Summry
This repository is the demo repository for T-Mech submission, "Origami-inspired wearable robots for trunk support"

## Installnation
We have included pynamics as a submodule (commit bcf35b8) so there is no need to install pynamics seperately, Please use the pynamics module included in the submodule as further version of pynamics might not support these demos. Other dependencies required: sympy, numpy, scipy, matplotlib, os, sys. We recommend run these code in Spyder IDE and python 3.8.

## Description
### Experiments simulation
Two code are used to repliacte the experiments:
  1. four_bar_2_exp.py is the code we used in Sec. III-D and Sec. IV-C. This code also generates Fig. 9.
  2. triangle_exp.py is the code we used in Sec. III-D and Sec. IV-C. This code also generates Fig. 10
These two code were used to simulate the experiments only and not used towards the system calculation.
### System calculation
One master code is used to generate the system states.
Two slave code are used to calculate the system dimension, namely, four_bar2_system.py and fourbar_with_two_triangle.py

## Contact
Please contact dongting@asu.edu for any questions.
