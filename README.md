# <img src="NAISRicon.svg" width="35" height="35"> NAISR: A 3D Neural Additive Model for Interpretable Shape Representation

Pytorch implementation for our `NAISR` paper <br>
[NAISR: A 3D Neural Additive Model for Interpretable Shape Representation](https://arxiv.org/abs/2303.09234), ICLR 2024 **Spotlight**. <br>
Yining Jiao, Carlton Zdanski, Julia Kimbell, Andrew Prince, Cameron Worden, Samuel Kirse, Christopher Rutter, Benjamin Shields, William Dunn, Jisan Mahmud, Marc Niethammer. <br>
UNC-Chapel Hill

#### [Paper](https://arxiv.org/abs/2303.09234) | [Project Page](https://uncbiag.github.io/NAISR/) | [Colab Demos](https://colab.research.google.com/drive/1OudGynEydIXpAgfA9lvi5d0L86kK-uwg?usp=sharing)


<div align=center><img src="https://github.com/uncbiag/NAISR/blob/main/figs/NAISRoverviewwide.svg?raw=true" width="1000" height="600"/></div>

Please cite as:
```
@inproceedings{
jiao2024naisr,
title={\texttt{NAISR}: A 3D Neural Additive Model for Interpretable Shape Representation},
author={Yining Jiao and Carlton Jude ZDANSKI and Julia S Kimbell and Andrew Prince and Cameron P Worden and Samuel Kirse and Christopher Rutter and Benjamin Shields and William Alexander Dunn and Jisan Mahmud and Marc Niethammer},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=wg8NPfeMF9}
}
```

>[!NOTE]
>For the three datasets used in `NAISR` paper, we provided this [Colab demo](https://colab.research.google.com/drive/1OudGynEydIXpAgfA9lvi5d0L86kK-uwg) to investigate the shapes learned with `NAISR`. <br>
We also provide the instructions on this page to apply `NAISR` to your customized shape analysis questions. 


## Installation
The code is tested with ``python=3.9``, ``torch=2.1.0``, ``torchvision=0.15.2``.
```
git clone https://github.com/uncbiag/NAISR
cd NAISR
```
Now, create a new conda environment and install required packages accordingly.
```
conda create -n naisr python=3.9
conda activate naisr
pip install -r requirements.txt
```


## Data and Model Weights
We train and test our method on three datasets: Starman, ADNI Hippocampus, and Pediatric Airways.

| Dataset   |                      Description             |           Dataset Link              |   Model Link              |
|-----------|----------------------------------------------|:------------------------------------:|:------------------------------------:|
|Starman      |  simulated 2D Starman shapes      |  [simulation code)][StarmanData]          | [weights][StarmanModel]    | 
|ADNI Hippocampus  |  1632 hippocampus from ADNI     |  [official site][ADNIData]              | [weights][ADNIModel]  |
|Pedeatric Airway |  357 pediatric airway shapes      |  NA | [weights][AirwayModel]               |



[ADNIData]: https://ida.loni.usc.edu/login.jsp?project=ADNI
[StarmanData]: 

[ADNIModel]: https://github.com/uncbiag/NAISR/releases/download/naisr_weights_v0/naisr_weights_adni.pth
[StarmanModel]: https://github.com/uncbiag/NAISR/releases/download/naisr_weights_v0/naisr_weights_starman.pth
[AirwayModel]: https://github.com/uncbiag/NAISR/releases/download/naisr_weights_v0/naisr_weights_pediatric_airway.pth


## Visualizations of Shape Space Extrapolation 

### Getting Started
One just need to use this [Colab demo](https://colab.research.google.com/drive/1OudGynEydIXpAgfA9lvi5d0L86kK-uwg) to explore the template shape space of {starmen, hippocampi and airways} used in the `NAISR` paper. 

### More functions
Our code repo provides more functions to visualize the shape space, e.g., as a matrix for a specific case or for the template shape (as Figure 3 in the main paper),

To get the template shape extrapolation, pls use
```
python evolution_shapematrix.py -e examples/hippocampus/naigsr_0920_base.json
```

To get the template shape extrapolation, pls use
```
python evolution_shapematrix_specific.py -e examples/hippocampus/naigsr_0920_base.json
```



## Customize
One may also want to use `NAISR` on their own shape analysis problems. For this use we provide our best suggestions/instructions here through the illustration for the simulated starman dataset,

### Data Preprocessing

#### Alignment with Rigid Transformation
The shapes to explore need to be registered with a rigid transformation (translation + rotation). 
If paired point clouds are available, we recommend to use ; otherwise, we recommend to use ICP to register the point clouds. 
In our case, we use airway landmarks to learn the rigid transformation; and ICP algorithms to register the point clouds of the hippocampi.

#### SDF Extraction

###
```json
 {
  "Description" : [ "This experiment learns a shape representation for starman dataset." ],
  "Device":0, 
  "DataSource": {"train": "/home/jyn/NAISR/examples/starman/2dshape_train_with_temp.csv",
                 "test": "/home/jyn/NAISR/examples/starman/2dshape_test_with_temp.csv"},
  "Split": null,
  "Network": "DeepNAIGSR",
  "NumEpochs": 300,
  "LoggingRoot": "/playpen-raid/jyn/NAISR/log",
  "ExperimentName": "DeepNAIGSR_STARMAN3D_0222_256_base",

  "EpochsTilCkpt": 10,
  "StepsTilSummary": 1000,
  "UseLBFGS": false,
  "DoublePrecision": false,
  "CheckpointPath": "",
  "CodeLength": 256,

  "AdditionalSnapshots" : [ 50, 100, 200, 300, 400, 500 ],
  "LearningRateSchedule" : [
    {
      "Type": "Step",
      "Initial": 0.00005,
      "Interval": 1000,
      "Factor": 0.5
    },
    {
      "Type": "Step",
      "Initial":  0.001,
      "Interval": 1000,
      "Factor": 0.5
    }],
  "SamplesPerScene" : 750,
  "BatchSize": 64,
  "DataLoaderThreads": 4,
  "ClampingDistance": 1,

  "Articulation": true,
  "NumAtcParts": 1,
  "TrainWithParts": false,
  "Class": "starman",
  "Attributes": ["cov_1", "cov_2"],
   "TemplateAttributes":  {"cov_1": 0, "cov_2": 0},
  "Backbone": "siren",
  "PosEnc": false,
  "InFeatures": 2,
  "HiddenFeatures": 256,
  "HidenLayers": 6,
  "OutFeatures": 1,
  "Loss": {
    "whether_sdf": true, 
    "whether_normal_constraint": true, 
    "whether_inter_constraint": true, 
    "whether_eikonal": true,
    "whether_code_regularization": true}
```
### Training
For training, one need to use `train_atlas_3dnaigsr.py` with different networking settings indicated with the `json` files, e.g., for the starman dataset,
```
python train_atlas_3dnaigsr.py -e examples/starman/naigsr_0222_base.json
```

### Shape Reconstruction
For testing/reconstruction without covariates, one need to `reconstruct_atlas.py` with different network settings indicated with the `json` files, e.g., for the starman dataset.
```
python reconstruct_atlas.py -e examples/starman/naigsr_0222_base.json 
```

For testing/reconstruction with covariates, one need to `reconstruct_atlas.py` with different network settings indicated with the `json` files, e.g., for the starman dataset.
```
python reconstruct_atlas_with_cov.py -e examples/starman/naigsr_0222_base.json 
```


### Shape Transport
For shape transport without covariates, one need to `reconstruct_atlas.py` with different network settings indicated with the `json` files, e.g., for the starman dataset.
```
python transport_general.py -e examples/starman/naigsr_0222_base.json 
```

For shape transport with covariates, one need to `reconstruct_atlas.py` with different network settings indicated with the `json` files, e.g., for the starman dataset.
```
python transport.py -e examples/starman/naigsr_0222_base.json 
```


### Shape Evolution and Disentanglement

One just need to use this [Colab demo](https://colab.research.google.com/drive/1OudGynEydIXpAgfA9lvi5d0L86kK-uwg) to explore the template shape space of {starmen, hippocampi and airways} used in the `NAISR` paper. 

To get the template shape extrapolation, pls use
```
python evolution_shapematrix.py -e examples/starman/naigsr_0222_base.json 
```
```

To get the template shape extrapolation, pls use
```
python evolution_shapematrix_specific.py -e examples/starman/naigsr_0222_base.json 
```





More instructions on the way...





## Cite this work
```
@inproceedings{
jiao2024naisr,
title={\texttt{NAISR}: A 3D Neural Additive Model for Interpretable Shape Representation},
author={Yining Jiao and Carlton Jude ZDANSKI and Julia S Kimbell and Andrew Prince and Cameron P Worden and Samuel Kirse and Christopher Rutter and Benjamin Shields and William Alexander Dunn and Jisan Mahmud and Marc Niethammer},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=wg8NPfeMF9}
}
```
