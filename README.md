# NI2S_problem

Code for reproducing our paper "Suppressing the Endogenous Negative Influence Through Node Intervention in Social Networks."

---

To run the code, first type the following in your terminal:
```
pip install -r requirements.txt
```
then
```
python main.py cond_name
```
Note: 'cond_name' can be either `random`, `follower`, or `proximity`.

`main.py` computes the intervention target for our proposed algorithms and comparison algorithms under the conditions specified by 'cond_name'.
Note that, since GreedyMIOA and AdvancedGreedy take a long time to execute, we recommend that you compute them individually using `compute_AdvancedGreedy_only.py` and `compute_GreedyMIOA_only.py`.

After that, run a simulation in which intervention is performed on the $k$ nodes selected by each algorithm, and calculate the proportion of negative opinions.

The list of intervention targets and simulation results are saved in the `./results/graph_name/cond_name` directory.

---

If you use this code, please cite the following paper:
```
TBA
```
