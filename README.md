# RA-SecureMPC-NMAB

This contains the implementation of **Resource-Aware Secure Multi-Party Edge Computation Offloading**.

Our experiment is based on the CIFAR10 dataset:
https://www.cs.toronto.edu/~kriz/cifar.html

How to run the program:

1. Run Dataset_Prepare.py with specifying

---$q$: quantization level;

---$p$: finite field size

2. Run Dataset.py with specifying

---$M$: number of workers;

---$l$: dataset size;

---$Num_Episode_Train$: number of rounds

---$CPU_mode$: distribution of workers' CPU frequencies

3. Run Bandit.py

4. Run Baseline.py

5. Run NMAB.py

Note: NMAB.py is based on [Neural Contextual Bandits with UCB-based Exploration](https://github.com/uclaml/NeuralUCB.git).
