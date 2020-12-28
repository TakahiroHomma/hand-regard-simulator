# hand-regard-simulator
Infants about two months old watch the movement of their hands, which is called hand regard. It is considered that after experiencing hand regard, infants may recognize their own hands. This hand-regard-simulator creates a simple model of hand regard and simulates the process by which infants recognize their hands.

To deal with time-varying input and output resulting from movements of infant’s hands, a real-time recurrent learning (RTRL) algorithm is adopted and [RTRL software](http://www.bioinf.jku.at/software/rtrl/) is used.

The simulation results show that information about recognition of the modeled hands of the infant is stored in cell assemblies, which were self-organized. Cell assemblies appear during the phase of U-shaped developments of hand regard, and the configuration of the cell assemblies changes with each U-shaped development. Furthermore, movements like general movements appear during the phase of U-shaped developments of hand regard. 

# Installation
This program depends on GNU Compiler Collection (GCC) and parallelized by OpenMP. It took about 48hours to calculate the training on a PC with an old CPU (Intel Xeon E5-2650).
You can compile the source code (src/rtrl.c) and create an executable program rtrl;

```
$ gcc -o rtrl rtrl.c -O2 -lm -fopenmp -mcmodel=large
```

## Running

To train the network:
1. Copy input data file (data/train/rtrlpars.txt) for training to an appropriate directory. 
2. Change to that directory.
3. Run the above program rtrl in that directory;

```
$ ./rtrl
```

# Documentation
See the [documentation](./documentation.md) for how to run tests and display and analyze output results.

# Reference
[1] T. Homma: Hand Recognition Obtained by Simulation of Hand Regard. Front. Psychol. 9:729. doi: 10.3389/fpsyg.2018.00729 | https://doi.org/10.3389/fpsyg.2018.00729

[2] T. Homma: A modeling study of generation mechanism of cell assembly to store information about hand recognition. Heliyon, Volume 6, Issue 11, November 2020, e05347 DOI: https://doi.org/10.1016/j.heliyon.2020.e05347| https://www.cell.com/heliyon/fulltext/S2405-8440(20)32190-3

# License
hand-regard-simulator was created by modifying [Real Time Recurrent Learning Software](http://www.bioinf.jku.at/software/rtrl/) developed by Prof. Sepp Hochreiter and distributed under GPL v2. Therefore, hand-regard-simulator is also licensed under GPLv2. See LICENSE.txt for more details.


