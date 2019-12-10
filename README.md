# Dynamic-randomized-gossip
We study average consensus in mobile multi-agent systems via randomized gossip

## How to use
### Requirements
- Python 3.6
- Jupyter 1.0
- Matplotlib
- PIP _(not mandatory but recommended)_

### Installation
Either install Jupyter and Matplotlib via PIP:
```
git clone git@code.harvard.edu:flb979/Dynamic-randomized-gossip.git && cd Dynamic-randomized-gossip
pip install -r ./requirements.txt
```
Or manually via https://jupyter.org/install and https://matplotlib.org/3.1.1/users/installing.html

### Run
Open the jupyter notebook
```
jupyter notebook
```
and within that notebook open any file ending on `.ipynb`, e.g.:

- `2019_oliva_fig3bcd.ipynb`, a reproduction of fig. 3 results from Oliva et al. (2019)
- `basic_scalings.ipynb`, some intuition for how decentralized average consesus performs under different parameters
- `communication_limits.ipynb`, average consensus performance for different numbers of simulatneously interacting agents
- `random_vs_gridgraph.ipynb`, a comparison of average consensus performance on random vs. grid graphs (Kilobot)
- `dcd_heuristics.ipynb`, a comparison of different heuristics for decentralized consensus detection (DCD)

Please run each cell individually! Sit back and watch the extravaganza!
