# PGM-DS: DBSCAN-based Particle Gaussian Mixture (PGM) Filters

## 1. Software Introduction

These are DBSCAN-based particle Gaussian mixture (PGM) filters:
There are two algorithms in this repository: PGM-DS and PGM-DU where DU utilised unscented transform (UT) in the updated step just as like as UKF.

Detail of the algorithm can be found from following paper:


**Developed by Sukkeun Kim, Cranfield University, UK**
* Email: <s.kim.aero@gmail.com>


## 2. Running the Demo

Just run with MATLAB! :)

## 3. Python Example (nonlinear dynamics)

*Py_PGM_DS_filters_nonlinear.py*: Only with PGM-DS and PGM-DU. The comparison with other filters and MC simulation are not considered in this code. All structure is the same as the Matlab code but just for the easy embedding for other environment.

## 4. Version Information

* 4th Sep 2024 Beta 0.0: First commit
* 23rd Oct 2024 Beta 0.1: Simple Python example added
* 24th Oct 2024 Beta 0.2: Parameter modification
* 6th Nov 2024 Beta 0.3: Minor function modification
* 11th Nov 2024 Beta 0.4: Updated figures
