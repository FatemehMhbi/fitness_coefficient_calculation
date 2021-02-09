# fitness_calculation
 
This script calculates fitness coefficient for clusters based on the abundance or frequency of each cluster through time. For instance the example frequency file, shows the counts of members of each cluster (clusters 0 and 1) every two weeks. The index of each row shows the two-weeks period (14 days apart from each other).
To minimize the sampling error, in hundred iterations the fitness coefficient of poisson distributed frequencies are obtained. Such that at each iteration, each value (abundance for each cluster at each timepoint) is replaced by the mean value of 2000 random samples drown from a poisson distribution using the value as expectation of interval.
And finally, the 95 percent confidence interval of fitness coefficient is reported for each of the clusters. The bigger the fitness coefficient, the more fitted the cluster.

Fitness coefficient calculation method comes from the following paper: \
Skums, Pavel, et al. "Numerical detection, measuring and analysis of differential interferon resistance for individual HCV intra-host variants and its influence on the therapy response." In silico biology 11.5, 6 (2011): 263-269.

# How to run:
Type in the command line: \
python3.7 sampling_error.py clusters_frequencies_example.csv \
It will save the result in a csv file adding "_CI_h_i" at the end of the frequency file name.
