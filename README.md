# SDCNDA
A Simple and Leakage-Free Relation-aware Contrastive Learning Framework for ncRNA–Disease Association Prediction

![SDCNDA](/image/Fig1.png)

## Introduction
In this work, we propose SDCNDA, a Simple and Leakage-Free Relation-aware Contrastive Learning Framework for ncRNA–Disease Association Prediction. SDCNDA reformulates ncRNA–disease representation learning as a biologically guided contrastive learning problem by integrating intra-type similarities and inter-type regulatory interactions into a heterogeneous graph and introducing relation-aware neighborhood masks to define contrastive supervision under distinct biological perspectives. Using a lightweight MLP-based encoder, SDCNDA learns unified node embeddings that preserve both semantic coherence within molecular types and regulatory consistency across biological layers. To ensure a fully rigorous and reproducible evaluation, all experiments are conducted under a strict leakage-free 5-fold cross-validation protocol: for each fold, 1/5 of the positive ncRNA–disease (and the corresponding miRNA–disease / miRNA–lncRNA) pairs are held out as the test split, the remaining edges are zeroed out of the target adjacency matrix before any similarity reconstruction or graph construction, and disease/miRNA/lncRNA similarities are reconstructed fold-wise via Gaussian GIP kernel and PBPA functional similarity so that no test-edge information can leak into the encoder. SDCNDA further adds no pseudo target edges and no path-compensation features, and the downstream classifier only consumes pair features built from the trained embeddings. Extensive experiments on multiple benchmark datasets and three prediction tasks (MDA / LDA / LMI) demonstrate that SDCNDA consistently outperforms state-of-the-art methods in predictive accuracy and efficiency, with low fold-to-fold variance under 5-fold CV, while case studies further confirm its ability to recover biologically plausible ncRNA–disease associations.

## RUN SDCNDA
### Requirements
The experiments are conducted in the following environment:
`Python 3.9.19` `PyTorch 2.0.0` `CUDA 11.8` `Numpy 1.26.3` `Pandas 1.4.4` `scikit-learn 1.2.2` `torch-geometric 2.5.3` `xgboost 3.1.3` `dgl 2.1.0`
Install the dependent python libraries by:
```
pip install -r Code/requirements.txt
```

### Data Preparation
The datasets used in our code are shown in 'SDCNDA/data/'. If you want to use your own dataset, the original data need to be preprocessed for the following work, prepare the following data:
```
data:
    --mi_dis.txt              
    --lnc_dis.txt            
    --dis_lnc.txt             
    --dis_sem_sim.txt
    --miRNA-names.txt / lncRNA_name.txt / disease_name.txt
```

### Data Split
After preparing the data, you need to split the data. Run `Code/build_fold_split.py` to get 5-fold train/test pairs (no test-edge leakage into encoder):
```
data:
    --mi_dis_train_id{1..5}.txt
    --mi_dis_test_id{1..5}.txt
    --lnc_dis_train_id{1..5}.txt
    --lnc_dis_test_id{1..5}.txt
    --mi_lnc_train_id{1..5}.txt
    --mi_lnc_test_id{1..5}.txt
```

### Graph Conversion and Representation Learning
After splitting the data, you need to convert the data structure and learn node embeddings. Run `Code/run_embedding.py` to reconstruct similarities fold-wise and train the graph contrastive encoder so as to obtain the per-node embeddings:
```
python Code/run_embedding.py --dataset dataset1 --task MDA --gpu 0 --contrastive-epochs 500
python Code/run_embedding.py --dataset dataset1 --task LDA --gpu 0 --contrastive-epochs 500
python Code/run_embedding.py --dataset dataset1 --task LMI --gpu 0 --contrastive-epochs 500

python Code/run_embedding.py --dataset dataset2 --task MDA --gpu 0 --contrastive-epochs 500
python Code/run_embedding.py --dataset dataset2 --task LDA --gpu 0 --contrastive-epochs 500
python Code/run_embedding.py --dataset dataset2 --task LMI --gpu 0 --contrastive-epochs 500
```

### Prediction
Run the following code (`Code/run_classifier.py`, default XGBoost; pass `--classifier mlp` to switch to the lightweight MLP variant) to get the predicting results:
```
python Code/run_classifier.py --dataset dataset1 --task MDA --embedding-run-name dataset1_mda
python Code/run_classifier.py --dataset dataset1 --task LDA --embedding-run-name dataset1_lda
python Code/run_classifier.py --dataset dataset1 --task LMI --embedding-run-name dataset1_lmi

python Code/run_classifier.py --dataset dataset2 --task MDA --embedding-run-name dataset2_mda
python Code/run_classifier.py --dataset dataset2 --task LDA --embedding-run-name dataset2_lda
python Code/run_classifier.py --dataset dataset2 --task LMI --embedding-run-name dataset2_lmi
```

### Or run all three steps end-to-end (XGBoost by default, MLP via `--classifier mlp`):
```
python Code/run_all.py --dataset dataset1 --tasks MDA LDA LMI
python Code/run_all.py --dataset dataset2 --tasks MDA LDA LMI
python Code/run_all.py --dataset dataset1 --tasks MDA LDA LMI --classifier mlp
python Code/run_all.py --dataset dataset2 --tasks MDA LDA LMI --classifier mlp
```