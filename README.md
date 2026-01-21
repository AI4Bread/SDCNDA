# SDCNDA
A Simple and Efficient Biologically Guided Dual Contrastive Learning Framework for ncRNA–Disease Association Prediction

![SDCNDA](/image/Fig1.jpg)

## Introduction
In this work, we propose SDCNDA, a Simple yet efficient biologically guided Dual Contrastive learning framework for NcRNA–Disease Association prediction. SDCNDA reformulates ncRNA–disease representation learning as a biologically guided contrastive learning problem by integrating intra-type similarities and inter-type regulatory interactions into a heterogeneous graph and introducing relation-aware neighborhood masks to define contrastive supervision under distinct biological perspectives. Using a lightweight MLP-based encoder, SDCNDA learns unified node embeddings that preserve both semantic coherence within molecular types and regulatory consistency across biological layers. Extensive experiments on multiple benchmark datasets demonstrate that SDCNDA consistently outperforms state-of-the-art methods in predictive accuracy and efficiency, while case studies further confirm its ability to recover biologically plausible ncRNA–disease associations.

## RUN SDCNDA
### Requirements
The experiments are conducted in the following environment:
`Python 3.9.23` `PyTorch 2.1.0` `CUDA 11.8` `Numpy 1.26.4` `Pandas 2.3.1` `matplotlib 3.9.4` `scikit-learn 0.24.2` `seaborn 0.13.2` `torch-geometric 2.6.1` Install the dependent python libraries by:
```
pip install -r requirements.txt
```


### Data Preparation
The datasets used in our code are shown in 'SDCNDA/data/'. If you want to use your own dataset, the original data need to be preprocessed for the following work, prepare the following data:
```
data:
    --mi_dis.txt
    --lnc_dis.txt
    --mi_lnc.txt
    --mi_fusion_sim.txt
    --mi_fusion_sim.txt
    --mi_fusion_sim.txt
```

### Data Split
After preparing the data, you need to split the data, run `split.py` to get:
```
data:
    --mi_dis_train_id.txt
    --mi_dis_test_id.txt
    --lnc_dis_train_id.txt
    --lnc_dis_test_id.txt
    --mi_lnc_train_id.txt
    --mi_lnc_test_id.txt
```

### Graph Conversion
After splitting the data, you need to convert the data structure, run `process.py` to get:
```
graph structure data:
    --edges.pt
    --feature.pt
    --label.pt
    --matrix_A.npy
    --samplefea.npy
```

### Representation learning
Run the following code to get the node embeddings:
```
python main.py
```

### Prediction
Run the following code to get the predicting results:
```
python MDA.py
python LDA.py
python LMI.py
```
