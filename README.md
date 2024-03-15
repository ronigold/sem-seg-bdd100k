
# Semantic Segmentation on BDD100K Dataset

## Overview
This project applies semantic segmentation techniques to the [BDD100K dataset](https://doc.bdd100k.com/index.html). 

We utilize the BDD100K dataset, known for its diverse driving conditions, including 10K images with rich annotations for tasks like semantic segmentation. The annotations are adjusted so that the pixel value 255, originally representing unknown objects, is mapped to 19 to fit our model's classification schema.

![Semantic Segmentation Samples](src/train_samples.png)

## Dataset

Below is the mapping from class IDs to their respective classes:

- `0`: Road
- `1`: Sidewalk
- `2`: Building
- `3`: Wall
- `4`: Fence
- `5`: Pole
- `6`: Traffic Light
- `7`: Traffic Sign
- `8`: Vegetation
- `9`: Terrain
- `10`: Sky
- `11`: Person
- `12`: Rider
- `13`: Car
- `14`: Truck
- `15`: Bus
- `16`: Train
- `17`: Motorcycle
- `18`: Bicycle
- `19`: Unknown
  
To download the datasets for the project please click on the following links:
- [Train dataset](https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip).
- [Valid dataset](https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip).
- [Test dataset](https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip).
- [Labels (for Train & Valid))](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip).

To successfully run the notebook, make sure you set your project's root path to:

![Tree Files](src/files_tree.png)

## Model
The project employs a U-Net architecture with a ResNet34 backbone, designed for precise pixel-wise classification across the dataset's 20 classes.

![U-Net with a ResNet34 backbone](src/unet.png)

The weights are pre-trained weights on ImageNet to optimize learning and exploit the potential of transfer learning

## Notebook
The project is encapsulated in a Jupyter Notebook, providing an end-to-end walkthrough from data preprocessing, model training, to evaluation and visualization of the results.

### How to Use
1. Clone the repository to your local machine or a compatible Jupyter environment.
2. Ensure you have Jupyter Notebook or JupyterLab installed.
3. Navigate to the cloned repository and launch the notebook:
   
```bash
jupyter notebook Training and evaluation.ipynb
```
4. Follow the steps in the notebook to train the model and evaluate its performance.

## Installation
To install the required libraries, run:

```bash
pip install -r requirements.txt
```
This will install all necessary dependencies to run the notebook, including PyTorch, FastAI, and PIL.

## Results
The notebook includes detailed sections on model evaluation, showcasing accuracy, Intersection over Union (IoU), and F1-score among other metrics. Visualizations are provided to compare the model's predictions against the ground truth.

The visual results shown below are based on the first model with the higher overall performance (cross_entropy)

### Sample from the Validation set (actual VS predicted)
![Cross Entropy Loss](src/valid_samples.png)

### Sample from the Test set
![Cross Entropy Loss](src/test_samples.png)

### Cross Entropy Loss

The analysis of the results shows that although the general results are not bad, there are degenerate classes. These classes are represented by missing representation in the data as can be seen:

#### Cross Entropy loss
![Cross Entropy Loss](src/cross_entropy_loss.png)

#### Segmentation 
![Segmentation Results CE](src/cross-entropy-seg-res.png)
![IOU VS PER. CE](src/ios-vs-perc-cross-entropy.png)


#### Classification
We can also use the model we built as a classification model.

The rule will be that if the image has at least one pixel that belongs to a certain class, that class appears in the image.

In this way it is possible to examine the performance of the model as a multi-label classification model:


![Confusion Matrix CE](src/confusion_matrix_cross_entropy.png)
![Classification Results CE](src/cla_res_cross_entropy.png)

### Focal Loss + Class Weights

The "Focal Loss + Class Weights" experiment is designed to address the challenges of training a deep learning model on an unbalanced dataset, where some classes are significantly underrepresented compared to others. Focal loss is an advanced loss function that modifies the standard cross-entropy loss to put more focus on hard, misclassified examples and less on easy examples. This is achieved by adding a modulating factor to the cross-entropy loss, which reduces the loss contribution from easy examples and increases the importance of correcting misclassified ones.

Incorporating class weights into the focal loss further enhances the model's ability to deal with class imbalance. Class weights are used to assign more significance to rare classes and less to common ones during the training process. By combining focal loss with class weights, the model is encouraged not only to focus on hard examples but also to pay more attention to underrepresented classes, thereby improving overall model performance on imbalanced datasets.

#### Focal Loss + Class Weights loss
![Cross Entropy Loss](src/focal_loss.png)
The analysis of the results shows that although the general results are not bad, there are degenerate classes. These classes are represented by missing representation in the data as can be seen:

#### Segmentation 
![Segmentation Results CE](src/focal-seg-res.png)
![IOU VS PER. CE](src/ios-vs-perc-focal.png)

#### Classification
![Confusion Matrix CE](src/confusion_matrix_focal.png)
![Classification Results CE](src/cla_res_focal.png)

#### Visualization (on Test Set)
![Confusion Matrix CE](src/test_set_samples_focal.png)

#### Conclusion
In summary, addressing data imbalance through adjustments in the loss function allowed the model to predict a broader range of classes (all the classes except train class). However, this approach led to a decrease in overall performance and introduced some noise. This issue likely stems from insufficient data for the underrepresented classes, making it difficult for the model to discern clear patterns. Consequently, this not only impacts the model's ability to learn from these classes but also affects its performance across all other classes.

## Semantic Search CLIP

In this section, we will outline a methodology for creating a pipeline capable of processing a query related to the content depicted within a database of images, and subsequently generating a pertinent response accompanied by the appropriate image. This response will be formulated in natural language and will be based on the semantic interpretation of the visual content identified in the image. This pipeline will exclusively utilize open-source models.

The construction of this pipeline necessitates the integration of two primary tools:

1. **CLIP (Contrastive Language–Image Pretraining)**: This model facilitates the embedding of images and text within a unified representational space, enabling the identification of images most closely aligned with the query posed by the user.

2. **LLaVA (Language and Vision Assistant)**: As a multimodal model, LLaVA is adept at processing images and generating descriptive narratives of the visuals contained therein.

Subsequent section will demonstrate how, by employing CLIP to create a vectorized database of our image collection, we can accurately retrieve images by specifying key visual elements through natural language queries. In this segment, we use CLIP (Contrastive Language–Image Pre-training) inversely from its common application. Rather than generating text from images, we aim to match textual prompts with a set of pre-processed image thumbnails. For each text prompt, CLIP's cross-modal capabilities are utilized to find the most relevant image from the dataset. 

The process involves preparing embeddings for all images in DBB100K to and then calculating the similarity between the embeddings of text prompts and images. Below, the results illustrate the closest image match for each prompt based on this methodology.

![CLIP Image Search Results](src/CLIP_res.png)

The subsequent segment delineates the operational framework and outcomes facilitated by the pipeline. The process is delineated as follows:

1. Initially, the pipeline transforms the entirety of the image database into vector representations utilizing the CLIP model.
2. Upon receiving a specific query, a technique akin to few-shot learning is employed to instruct the LLaVA model on extracting pertinent anchors from the query. These anchors serve as the basis for identifying the relevant image in the subsequent step.
3. The extracted anchors are then vectorized, enabling the identification of the image vector that most closely aligns with these anchor vectors.
4. Following the identification of the image that corresponds to the anchors present in the initial query, the pipeline revisits LLaVA, requesting it to formulate a response to the user's original question, referencing the identified image.
5. The process culminates in the presentation of both the located image and LLaVA's generated response within the user interface, offering a comprehensive answer to the user's query.

![CLIP Image Search Results](src/side_by_side_chats.png)

### How to Use
1. Clone the repository to your local machine or a compatible Jupyter environment.
2. Ensure you have Jupyter Notebook or JupyterLab installed.
3. Navigate to the cloned repository and launch the notebook:
   
```bash
jupyter notebook Semantic Search CLIP.ipynb
```

## Contributing
Contributions are welcome. If you have suggestions for improving the project, please open an issue or submit a pull request.

## License
Using the BDD100K dataset requires compliance with its license terms. Ensure to adhere to these when utilizing the dataset.

## Contact
For questions or support, feel free to open an issue in the repository. Your feedback and inquiries are highly appreciated.

