### The source code is being organized.
----

# Select and Optimize: Learning to Solve Large-Scale Traveling Salesmen Problem
![Pipeline](evaluate/results/Pipeline.png)
Our *select-and-optimize* iterative framework: At each step, batch sub-problems are sampled and then fed into our selector-and-optimizer network. We select the most promising sub-problem and transform it into a new sub-solution $X_s$ by our network. The destroy-and-repair method is applied on the new solution $X'$ at a certain updating interval.

## Dependencies
* Python>=3.8
* NumPy
* sklearn
* elkai
* [PyTorch>=1.2](http://pytorch.org/)
* tqdm
* Matplotlib (optional, only for plotting)

## Data Generation
### Train/Test set generation: 
```bash
cd evaluate
python generate_tsp.py --n_points 200 --num_samples 1000 --mode train 
python generate_tsp.py --n_points 2000 --num_samples 32 --mode test 
```
### Initialization
* first step: compile LKH heuristic
```bash
cd evaluate
make
```
* scale_type: medium/large/tsplib
```bash 
cd evaluate
python initialization.py --n_points 200 --scale_type medium
python initialization.py --n_points 2000 --scale_type large
```
### Sub-problems training set generation: 
```bash
cd evaluate
python generate_sub_problem.py --n_points 200 --mode train
cd ..
cp evaluate/data/train/train200_subproblems.pkl train/TSP/POMO/data/tsp/
```

## Training
```bash
cd train/TSP/POMO/ 
python train.py
```
In the train.py file
* Train SO100 model, set env_params = {
    'problem_size': 100,
    'data_file': 'data/tsp/100_train_subproblem.pkl'}
* Train SO50 model, set env_params = {
    'problem_size': 50,
    'data_file': 'data/tsp/50_train_subproblem.pkl'}

## Evaluation
### Quick start
* cd evaluate, run select_and_optimize.ipynb on Jupyter notebook
### Initialization
* scale_type: medium/large/tsplib
```bash 
cd evaluate
python initialization.py --scale_type tsplib
```
### select-and-optimize evaluation
* medium-scale data:
```bash
cd evaluate
use SO100: python popmusic_pomo.py --n_points 200 --T 50 --scale_type medium --init_enable True --reverse_enable True --if_query_tail True
use SO50: python popmusic_pomo.py --n_points 200 --T 50 --scale_type medium --init_runs 50 --sub_pro_size 50 --reverse_enable True --if_query_tail True
```
* large-scale data:
```bash
cd evaluate
use SO100: python popmusic_pomo.py --n_points 2000 --T 60 --scale_type large --init_enable True --reverse_enable True --if_query_tail True --dr_enable True
use sO50: python popmusic_pomo.py --n_points 2000 --T 60 --scale_type large --sub_pro_size 50 --reverse_enable True --if_query_tail True --dr_enable True
```

* TSPLIB data:
 ```bash
cd evaluate
use SO100: python popmusic_tsplib.py --T 100 --init_enable True --reverse_enable True --if_query_tail True --tsplib_path data/tsplib_large_new/
use SO50: python popmusic_tsplib.py --T 100 --reverse_enable True --if_query_tail True --sub_pro_size 50 --tsplib_path data/tsplib_large_SO100/
 ```



## Ablation study
* selector comparision, sample method: random/count/nn_random
```bash
cd evaluate
python popmusic_pomo.py --n_points 200 --T 50 --scale_type medium --init_enable True --reverse_enable True --if_query_tail True --sample_method random
```
* without tail embedding: 
```bash
cd evaluate
python popmusic_pomo.py --n_points 2000 --T 50 --sub_pro_size 100 --init_enable True --reverse_enable True 
```
* without exploiting symmetry: 
```bash
cd evaluate
python popmusic_pomo.py --n_points 2000 --T 50 --sub_pro_size 100 ---init_enable True --if_query_tail True
```
* without destroy-and-repair method: 
```bash
cd evaluate
python popmusic_pomo.py --n_points 2000 --T 100 --sub_pro_size 100 --init_enable True --reverse_enable True --if_query_tail True
```


# Acknowledgement
The code are based on the repos [POMO](https://github.com/yd-kwon/POMO)

