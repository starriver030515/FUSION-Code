## Format

Inspired by LLaVA-OneVision, we use YAML files to configure our training datasets. A sample YAML configuration is shown below:

```
datasets:
  - json_path: ../../playground/train/jsons/LLaVA-Pretrain-558k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/ShareCaptioner-1246k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/URSA_Alignment-860k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthColor-244k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthCount-300k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthScene-268k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthSpatial-300k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/SynthTextVQA-400k.json
    sampling_strategy: "all"
  - json_path: ../../playground/train/jsons/PixelProse-5500k.json
    sampling_strategy: "all"
```

You can adjust the sampling_strategy for each dataset to control the size and proportion of the data used during training. For details on how sampling works, please refer to the implementation in [fusion/train/train.py](../fusion/train/train.py).

Our official training YAML files for FUSION are located in [scripts/train/yaml](../scripts/train/yaml). You can modify these files to suit your training needs.