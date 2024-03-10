
# Semantic Segmentation on BDD100K Dataset

## Overview
This project applies semantic segmentation techniques to the BDD100K dataset. The goal is to classify pixel-wise annotations to understand the driving environment better.

![Semantic Segmentation Samples](bdd100k_seg.png)

## Dataset
The dataset used is the BDD100K, which provides 10K images with rich annotations for various tasks including semantic segmentation. Annotations have been adjusted so that the pixel value `255` corresponds to the `unknown` class and is represented by `19` in our adjusted masks.

## Model
We use a U-Net architecture with a ResNet34 backbone for the task of semantic segmentation. The model is trained to predict 20 classes based on the BDD100K dataset specification.

## Installation
Instructions for setting up the project environment:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Usage
To train the model, execute the following command:

```bash
python train.py --data_path /path/to/dataset
```

To evaluate the model and generate segmentation masks:

```bash
python evaluate.py --data_path /path/to/dataset --model_path /path/to/model
```

## Custom Scripts
- `get_adjusted_mask_file_path`: A custom function included in the data preprocessing step that adjusts mask annotations for training.

## Results
After training, the results are evaluated on a hold-out test set. Metrics such as accuracy, IoU, and F1-score are reported.

## Contributing
Contributions to the project are welcome. Please fork the repository and submit a pull request.

## License
By using the BDD100K dataset, you agree to comply with the BDD100K license terms.

## Contact
For questions and support, please open an issue in the repository.


