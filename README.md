# Improving the efficiency of CMA-ES as a global optimization algorithm

This repository contains all the code behind my Bachelor's Thesis. Quick breakdown of the structure:

* `lib` - the framework for running and analyzing experiment results (`metrics.py`, `metrics_collector.py`, `enums.py`), optimizer definitions (`optimizers.py`), benchmark function wrappers (`funs.py`) and a bunch of other reusable snippets.
* `experiments` - all experiments described in the thesis with a few extra ones that didn't make the cut, but provided insight regarding research directions. Most notably:
    * `experimentum_crucis/` - chapter 3.1, verification and validation of the claims of CMA-ES' C matrix converging towards the inverse Hessian of the quadratic objective function
    * `quad_convergence` - chapter 3.2, analysis of the relationship between BFGS convergence speed and the initial inverse Hessian approximation
    * `find_switch_interval/` - the code behind chapters 3.3 & 3.4 - CMABFGS convergence and ECDF curves on various benchmark functions with varying search bounds
    * `manuscript_visualization` - some quick scripts to create a plot or two that don't require heavy calculation on a HPC, some of these weren't used, these should probably end up in `scripts/`
    * the other subdirectories contain code behind legacy experiments, possibly not compatible with the `lib` interfaces that changed over time. These shaped the research direction in some way, but weren't described in the thesis
* `scripts/` - this was supposed to contain code that didn't require heavy calculation to generate results, just some preliminary quick experiments

All of the experiments are runnable via the PLGrid cluster using the provided sbatch scripts, configured by `config.txt` files in their `slurm` subdirectories. As only `experimentum_crucis` and `find_switch_interval` were used in the thesis at the end of the day, their interface is more or less unified while others may vary.

## Reproducing the thesis results

This section serves as a pointer on how to run the experiments that were used in the final manuscript along with data processing and plot generation. These instructions are specific to running them via slurm on the PLGrid cluster.

### Chapter 3.1 (Experimentum crucis)

#### Configuration file format (`experiments/experimentum_crucis/config.txt`)

Three columns are to be present:
- the dimensionality
- number of runs to average the results from
- how often the results are to be saved for processing (the output files can get pretty large, so the granularity of the datapoints is parametrized)

Example:
```
10   25  1
20   25  1
50   25  1
100  25  4
```

This corresponds to four independent configurations - in 10, 20, 50 and 100 dimensions, averaging from 25 runs, where for $d<100$, the metrics are saved after every CMA-ES iteration and for $d=100$ they're saved every four iterations.

#### Running the experiment
From the project root, just run the script:
`./experiments/experimentum_crucis/run_many.sbatch`

A job array with a job per configuration will be ran.

#### Plot generation
Since in this case the visualization doesn't require much compute power, just run the `experimentum_crucis/visualize.py` script on your local machine or on the cluster. The specific dimensionalities to process are defined inline to keep it simple. Note that a lot more metrics are generated and visualized than those reported in the manuscript - some are more sensible (and were left in), some add no extra insight (and were left out). 

### Chapter 3.2 (BFGS Acceleration)

#### Configuration file format (`experiments/quad_convergence/config.txt`)

Four columns are to be present:
- the dimensionality
- number of runs to average the results from
- how often to collect and save CMA-ES covariance matrices (in CMA-ES iterations)
- after which iterations to run BFGS instances for later comparisons, separated by dashes

Example:

```
10   25  20  20-40-60-80-100-120-140-160-180-200
100   25  100  200-400-600-800-1000-1200-1400-1600
```

Two configurations, both with 25 runs averaged. One in 10 dimensions, with the covariance matrix saved every 20 CMA-ES iterations and BFGS instances ran every 20 CMA-ES iterations, starting at 20 and ending at 200. The other one in 100 dimensions, saving the matrix every 100 iterations and running BFGS every 200 iterations from 200 to 1600. 

#### Running the experiment
From the project root, just run the script:
`./experiments/quad_convergence/run_many.sbatch`

#### Plot generation

> TODO



### Chapter 3.3 & 3.4 (CMABFGS benchmarking)
> TODO


### Note on the state of the repository

As this work is in progress and the thesis is not supposed to be its culminating point, the repository was not cleaned up to serve as an archive since I don't want to delete any older/partial results and experiments to make it look pretty. They might be useful in the future and I see no point in cleaning up what was already once done and might be used in the future after a slight rework. Due to this, it's like every other research repository - some things don't work, some are abstracted poorly, some should be refactored.
