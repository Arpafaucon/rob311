# ROB311 - 3 : Partially Observable Markov Decision Process

Grégoire ROUSSEL

1st october 2018

## Description

This TP illustrates the process of robot localisation using its observation. We assume to be in the case of a Partially Observable Markov Decision Process, therefore that there is a hidden Markov Model we only get partial information on.

### Status
This TP is working and has been commented.


## Usage Example
```sh
python3 main.py
```
Standard output
```
case: Localized
Starting POMDP with
0.05 0.1 0.1 0.7 0.005 0.01 0.005 0.03


>>> Step 1 (action=R, observed=L)
0.005 0.011 0.006 0.004 0.005 0.005 0.959 0.005
>>> Step 2 (action=D, observed=U)
0.051 0.001 0.051 0.001 0.051 0.055 0.008 0.782


case: Uniform Dist
Starting POMDP with
0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125


>>> Step 1 (action=R, observed=L)
0.019 0.056 0.019 0.009 0.019 0.019 0.815 0.046
>>> Step 2 (action=D, observed=U)
0.058 0.001 0.058 0.001 0.058 0.073 0.008 0.743

```

### Results analysis
We observe that in both cases, there is a fast convergence of the belief towards one priviledged state, which is the only one possible. Indeed, the sequence of action Right->Down is only possible with the sequence of states 4-7-8.
In case 1, the initial belief is already pointing to the right state, 4, and one step later there is a strong confidence to be in state 7 (as we do expect when looking at the model).
In case 2, with an uniform starting distribution, the belief already very densely focused on state 4 at step 1.
This should be linked to the unique position of the `L` feature.


`NOTE` : there are color codes in the display to help understanding. For example, in state vectors, the biggest values are colored to stand out of the others.

