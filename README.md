# recommended system models implemented with tensorflow
## 1. requirement and environment
### Python 3.6
### Tensorflow 1.12-gpu
### Scikit-learn 0.23.0
### numpy 1.18.1

## 2. ctr/cvr models
### 2.1) full-connect-network
#### instance format: 
#### cd dnn && python train_dnn.py 20200520 20200523
### 2.2) deep-structured-semantic-match-network
#### instance format:
#### cd dssm && python train_dssm.py 20200520 20200523
### 2.3) deep-match-prediction-network
#### instance format:
#### cd deepmp && python train_deepmp.py 20200520 20200523
### 2.4) deep-match-rank-network
#### instance format:
#### cd deepmr && python train_deepmr.py 20200520 20200523
### 2.5) deep-interest-network
#### instance format:
#### cd din && python train_din.py
