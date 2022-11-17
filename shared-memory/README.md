---
author:
- Will Querée
bibliography:
- references.bib
date: November 2022
title: |
  **Shared Memory Coursework Report**\
  CM30225
---

Introduction
============

This coursework seeks to implement, in parallel, the relaxation
technique for solving differential equations. Repeating the definition
in the problem statement, this takes all non-boundary elements in an
$n\times n$ matrix, and sets their values to the average of their 4
neighbours. The process is iteratively repeated until every value is
within some precision, $\theta$, of the previous iteration. This means
that performing one iteration of this method on $A$ results in $A'$, as
shown in Equations [\[eqn1\]](#eqn1){reference-type="ref"
reference="eqn1"} and [\[eqn2\]](#eqn2){reference-type="ref"
reference="eqn2"}.

$$\label{eqn1}
A=\begin{bmatrix}
1.00 & 0.00 & 0.00 & 0.00\\
1.00 & 0.00 & 0.00 & 0.00\\
1.00 & 0.00 & 0.00 & 0.00\\
1.00 & 1.00 & 1.00 & 1.00\\
\end{bmatrix}$$

$$\label{eqn2}
A'=\begin{bmatrix}
1.00 & 0.00 & 0.00 & 0.00\\
1.00 & 0.25 & 0.00 & 0.00\\
1.00 & 0.50 & 0.25 & 0.00\\
1.00 & 1.00 & 1.00 & 1.00\\
\end{bmatrix}$$

The solution will be developed using the C programming language, with
the help of the `pthread`s and their associated parallelisation
primitives to ensure safe memory access.

Definitions
===========

For the purpose of complexity analysis in this report, the following
notation is defined:

-   $n$: The side length of the input matrix.

-   $N$: The total number of elements in the matrix ($n^2$).

-   $p$: The number of threads in use.

Approach
========

General Strategy
----------------

To make it possible to benchmark the parallel solution, in the first
instance, a fully serial implementation was developed. This used a `for`
loop to iterate over the \"inner\" elements in the matrix and replace
each of these with the average of its four neighbours. This will serve
as the ground truth method to measure the parallel implementation
against. The precision, $\theta$, is defined as `PRECISION` using a
preprocessor directive. The serial and parallel implementations are
submitted as `serial.c` and `parallel.c` respectively. A header file,
`utils.h`, is also implemented for functionality shared across the two
implementations.

The basis of the approach is a `while` loop which continues to iterate
until the value of the boolean `stop` evaluates to `true`. This occurs
when the value of each element in the main matrix is within $\theta$ of
its value in the previous iteration.

Correctness Testing
-------------------

To establish the correctness of my parallel implementation, a Python
script was created to randomly generate square, floating-point matrices.
It was then possible to call both the serial and parallel
implementations on the same data, write the outputs of each to a file,
and compare the resulting files using Python's built-in `filecmp`
library. The correctness of the serial implementation was established by
testing manually on some very small matrices. The script used to compare
the serial and parallel implementations is included in the submission as
`test-correctness.py`. Note that for this script to be run, NumPy
[@harris2020array] is required. This script should be saved in a
subdirectory of the project root. The project root should contain the C
source files and compiled binaries.

Benchmark Testing {#section:benchmark}
-----------------

The implementations were benchmarked using a randomly initialised
$2048\times2048$ matrix (this particular matrix was used consistently
across runs to ensure fairness). $n=2048$ was chosen as it would be
sufficiently large to test Speedup, $S_p$; and Efficiency, $E_p$, as
defined in Equations [\[eqn:Sp\]](#eqn:Sp){reference-type="ref"
reference="eqn:Sp"} and [\[eqn:Ep\]](#eqn:Ep){reference-type="ref"
reference="eqn:Ep"} respectively. To enable the use of these metrics,
the serial implementation was run for three repeats. This resulted in an
average observed elapsed time, $t_s=574.550$ s. $t_p$ is the observed
elapsed time on $p$ parallel processors. The values used for $p$ vary
according to the test in question. The effectiveness of the solution can
also be measured using the Karp-Flatt metric [@10.1145/78607.78614],
$e$, which measures the serial part of a computation. This is defined in
Equation [\[eqn:e\]](#eqn:e){reference-type="ref" reference="eqn:e"}.

$$\label{eqn:Sp}
S_p = \frac{t_s}{t_p}$$

$$\label{eqn:Ep}
E_p = \frac{t_s}{pt_p} = \frac{S_p}{p}$$

$$\label{eqn:e}
e = \frac{\frac{1}{S_p}-\frac{1}{p}}{1-\frac{1}{p}}$$

Scalability Testing
-------------------

The scalability of my final implementation was tested using the method
described in Section [3.3](#section:benchmark){reference-type="ref"
reference="section:benchmark"} across matrices for
$n\in\{256, 512, 1024, 2048\}$ and
$p\in\{1, 5, 10, 15, 20, 25, 30, 35, 40\}$.

Development
===========

Version 1
---------

### Strategy

The initial version of the parallel implementation sought to parallelise
the main update loop in the serial implementation, shown in Figure
[\[fig:serialloop\]](#fig:serialloop){reference-type="ref"
reference="fig:serialloop"}. This was achieved by splitting up the
elements in the matrix evenly (or as close to evenly as possible) into
$p$ batches. This will ensure that (provided threads are never blocked
from running) all threads are busy for approximately the same amount of
time. Then, it was possible to iterate over these chunks of the problem
as in the serial implementation to update the main matrix. This still
relies on taking a temporary copy of the main matrix. In order to ensure
that no concurrent writes could occur, I used a `pthread_mutex_t` to
guard access to the main matrix.

Test runs for Version 1 were completed using multiples of 20 for the
value of $p$ in the interval $[20, 160]$.

``` {.objectivec language="C"}
for (size_t i = 1; i < size - 1; i++) {
        for (size_t j = 1; j < size - 1; j++) {
            double meanValues[] = {
                copy[i - 1][j],
                copy[i][j + 1],
                copy[i + 1][j],
                copy[i][j - 1]
            };
            mat[i][j] = doubleMean(meanValues, 4);
            if (fabs(mat[i][j] - copy[i][j]) > PRECISION) {
                stop = false;
            }
        }
    }
```

### Analysis {#v1results}

The average elapsed time using $p$ threads is shown in Table
[1](#tab:timev1){reference-type="ref" reference="tab:timev1"}, and
measured in terms of Speedup, Efficiency, and Karp-Flatt in Table
[2](#tab:metricsv1){reference-type="ref" reference="tab:metricsv1"}.
This is visualised in Figures [1](#fig:Sp1){reference-type="ref"
reference="fig:Sp1"}, [2](#fig:Ep1){reference-type="ref"
reference="fig:Ep1"}, and [3](#fig:kf1){reference-type="ref"
reference="fig:kf1"}. It was evident from the Efficiency and Karp-Flatt
scores for this implementation that core utilisation was low. This
indicates that there was significant work being executed serially.
Investigation into the implementation revealed two bottlenecks.

The first bottleneck was due to the serial nature of the write
operation. That is, the main matrix was locked as a whole for the write
of an individual element. So, despite the fact that there were (assuming
maximal thread use) $p$ values being calculated simultaneously, the
locking of the main matrix in memory meant that only one value could be
written at once, resulting in $O(N)$ time complexity. Assuming perfectly
parallel speedup for the read operation, this would give a complexity
$O(\frac{N}{p} + N)$.

The second bottleneck related to the read operation, which was
implemented serially in its entirety from the outset. It is likely that
the overhead of this was non-trivial, and was dominating the run time.
Assuming perfectly parallel speedup for the write operation, this would
give an overall complexity $O(N + \frac{N}{p})$.

In addition to the limitations described above, it can be observed that
there is a significant dropoff in speedup after a peak using 40 threads.
It is likely that this dropoff is caused by the overhead of
orchestrating moving data between nodes, as the HPC cluster has a
maximum of 44 tasks per node. In cases where more than 44 threads are
used, this overhead may have been dominating the run time.

::: {#tab:timev1}
   **Threads ($p$)**   **Elapsed Time ($t_p$) / s**
  ------------------- ------------------------------
          20                     236.400
          40                     215.908
          60                     224.479
          80                     224.343
          100                    227.028
          120                    228.926
          140                    233.928
          160                    238.610

  : Mean Elapsed Time using $p$ Threads for Version 1
:::

::: {#tab:metricsv1}
   **Threads ($p$)**   **Speedup ($S_p$)**   **Efficiency ($E_p$)**   **Karp-Flatt Metric ($e$)**
  ------------------- --------------------- ------------------------ -----------------------------
          20                  2.439                  0.122                       0.357
          40                  2.670                  0.067                       0.349
          60                  2.568                  0.043                       0.372
          80                  2.570                  0.032                       0.376
          100                 2.540                  0.025                       0.384
          120                 2.519                  0.021                       0.389
          140                 2.465                  0.018                       0.399
          160                 2.416                  0.015                       0.408

  : Speedup, Efficiency, and Karp-Flatt Metrics using $p$ Threads for
  Version 1
:::

![Graph of Speedup using $p$ Threads for Version 1](Sp1.jpg){#fig:Sp1}

![Graph of Efficiency using $p$ Threads for Version
1](Ep1.jpg){#fig:Ep1}

![Graph of Karp-Flatt using $p$ Threads for Version
1](kf1.jpg){#fig:kf1}

Version 2
---------

### Strategy

It was conjectured in Section [4.1.2](#v1results){reference-type="ref"
reference="v1results"} that low efficiency and Karp-Flatt was due to
excessive serial computation in both the read and the write steps.

The first bottleneck was due to the serial nature of the write
operation. That is, the main matrix was locked as a whole for the write
of an individual element. There were two possible avenues to rectify
this, detailed below.

1.  Use a matrix of mutual exclusions - one for each element in the
    matrix that can be locked and unlocked for matrix accesses.

2.  Remove the use of mutual exclusions entirely, as the batching
    process I have implemented inherently ensures that no unsafe
    accesses can occur.

I chose option 2, as the overhead of constantly locking and unlocking is
both significant and unnecessary, for the reasons stated above.

The second bottleneck was more subtle. This arose from the serial nature
of the operation that creates a copy of the main matrix for later use in
the value calculation and write step. This was executing in $O(N)$ time
as it was not implemented in parallel. This was addressed in Version 2,
and is now parallelised using a similar batch-per-thread approach to
that of the write operation.

Test runs for Version 2 were only completed using values of $p$ as
multiples of 5 in the interval $(0, 40]$ to combat the efficiency
dropoff observed in testing for Version 1.

### Analysis {#analysis}

It can be observed from Tables [3](#tab:timev2){reference-type="ref"
reference="tab:timev2"} and [4](#tab:metricsv2){reference-type="ref"
reference="tab:metricsv2"}; and Figures
[4](#fig:Sp2){reference-type="ref" reference="fig:Sp2"},
[5](#fig:Ep2){reference-type="ref" reference="fig:Ep2"}, and
[6](#fig:kf2){reference-type="ref" reference="fig:kf2"} that the changes
of Version 2 effect a significant improvement in all metrics over
Version 1. The general trend shown by Figure
[4](#fig:Sp2){reference-type="ref" reference="fig:Sp2"} is a sharp
initial speedup for $1<p\leq20$, with speedup starting to plateau for
$p>20$. This is in line with what we would expect. Version 2 will serve
as a suitable baseline for scalability testing in Section
[5](#section:scalability){reference-type="ref"
reference="section:scalability"}.

::: {#tab:timev2}
   **Threads ($p$)**   **Elapsed Time ($t_p$) / s**
  ------------------- ------------------------------
           1                     784.600
           5                     242.012
          10                     131.896
          15                      59.797
          20                      47.601
          25                      52.028
          30                      45.366
          35                      39.739
          40                      51.523

  : Mean Elapsed Time using $p$ Threads for Version 2
:::

::: {#tab:metricsv2}
   **Threads ($p$)**   **Speedup ($S_p$)**   **Efficiency ($E_p$)**   **Karp-Flatt Metric ($e$)**
  ------------------- --------------------- ------------------------ -----------------------------
           1                  0.735                  0.735                        \-
           5                  2.382                  0.476                       0.170
          10                  4.371                  0.437                       0.118
          15                  9.642                  0.643                       0.032
          20                 12.112                  0.606                       0.030
          25                 11.081                  0.443                       0.049
          30                 12.709                  0.424                       0.044
          35                 14.509                  0.415                       0.040
          40                 11.190                  0.280                       0.064

  : Speedup, Efficiency, and Karp-Flatt Metrics using $p$ Threads for
  Version 2
:::

![Graph of Speedup using $p$ Threads for Version 2](Sp2.jpg){#fig:Sp2}

![Graph of Efficiency using $p$ Threads for Version
2](Ep2.jpg){#fig:Ep2}

![Graph of Karp-Flatt using $p$ Threads for Version
2](kf2.jpg){#fig:kf2}

Scalability {#section:scalability}
===========

From the runs shown in Figures [7](#fig:Spscale){reference-type="ref"
reference="fig:Spscale"}, [8](#fig:Epscale){reference-type="ref"
reference="fig:Epscale"}, and [9](#fig:kfscale){reference-type="ref"
reference="fig:kfscale"}, it can be observed that the effectiveness of
the developed solution improves with size, with the speedup from
parallelising the implementation maximised for $n=2048$. For $n=256$ and
$n=512$, Figure [7](#fig:Spscale){reference-type="ref"
reference="fig:Spscale"} shows speedup gradually tapering off above
$p\approx15$. This can be explained as past this point, the overhead of
orchestrating the parallelisation outweighs the benefit gained from the
parallelisation. For $n=1024$ and $n=2048$ however, the same cannot be
said. Here, initial maxima in speedup are observed at $p\approx20$, and
then again at $p\approx30$ for $n=1024$ and $p\approx35$ for $n=2048$.
This implies that unlike for the two smaller matrices, the benefit of
the parallelisation begins to outweigh the orchestration overhead for
larger and larger $p$.

![Graph of Speedup using $p$ threads for
$n\in\{256, 512, 1024, 2048\}$](Spscale.jpg){#fig:Spscale}

Unfortunately, it was not possible to test $n=4096$, as the serial run
exceeded the HPC cluster time limit of 20 minutes. Expectation would be
that for larger matrices, efficiency and speedup will increase, and
Karp-Flatt will decrease.

![Graph of Efficiency using $p$ threads for
$n\in\{256, 512, 1024, 2048\}$](Epscale.jpg){#fig:Epscale}

![Graph of Karp-Flatt using $p$ threads for
$n\in\{256, 512, 1024, 2048\}$](kfscale.jpg){#fig:kfscale}
