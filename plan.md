# Plan
1. Using hd5 datasets from ANN-Benchmarks
    - https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file
2. When using any shared/distributed method, all "processes" send their top k and then we find top k from a bunch of top k's
3. Figure out
    - Change in membership per iteration threshold - Track how recall changes with change in this threshold
    - Load balancing?
        - How many points per node?
    - How many clusters to create?
    - How many clusters to query?

---

# Resources
- https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Chandramohan-Fall-2012-CSE633.pdf

---

# OpenMP
- Cache locality

# MPI
- Communication

---

# Serial implementation of kNN
1. cmake and make
2. main.cpp file
    - Read the dataset
    - Call the knn cluster creation code
    - Verification
        - Query the above clusters
        - Verify neighbours
        - Calc recall
