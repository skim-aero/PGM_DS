# PGM-DS: DBSCAN-based Particle Gaussian Mixture (PGM) Filters

## 1. Software Introduction

This is a repository of DBSCAN-based particle Gaussian mixture (PGM) filters: PGM-DS and PGM-DU where DU utilised unscented transform (UT) in the updated step just as like as UKF.

Detail of the algorithm can be found from following paper:
[PGM-DS/DU](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5158079)

Note that the paper is based on Beta 0.5.


**Developed by Sukkeun Kim, Cranfield University, UK**
* Email: <s.kim.aero@gmail.com>


## 2. Running the Demo

Just run with MATLAB! :)

## 3. Python Example (nonlinear dynamics)

*PGM_DS_filters_nonlinear.py*: Only with PGM-DS and PGM-DU. The comparison with other filters and MC simulation are not considered in this code. All structure is the same as the Matlab code but just for the easy embedding for other environment.

## 4. Version Information

* 4th Sep 2024 Beta 0.0: First commit
* 23rd Oct 2024 Beta 0.1: Simple Python example added
* 24th Oct 2024 Beta 0.2: Parameter modification
* 6th  Nov 2024 Beta 0.3: Minor function modification
* 11th Nov 2024 Beta 0.4: Updated figures
* 14th Nov 2024 Beta 0.5: Minor modifications
* 16th Nov 2024 Beta 0.6: Minor function modification + Blid tryciclist example added
* 17th Nov 2024 Beta 0.7: Updated functions
* 18th Nov 2024 Beta 0.8: Updated functions
* 19th Nov 2024 Beta 0.9: Minor modifications
* 2nd  Dec 2024 Release 1.0: First release
* 17th Dec 2024 Release 1.1: Minor modifications
* 21st May 2025 Release 1.2: Simulation modifications
* 23rd May 2025 Release 1.3: SIR filter updated
* 26th May 2025 Release 1.4: Simulation modifications