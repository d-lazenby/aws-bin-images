# Object counting using the Amazon Bin Image Dataset (ABID)

Distribution centres and warehouses require careful inventory management to avoid over- or understocking, which can negatively affect customer experience, hinder efficiency and increase costs. <a href="https://landingcube.com/amazon-statistics/#:~:text=Amazon%20ships%20approximately%201.6%20million,and%2018.5%20orders%20per%20second" target="_blank" rel="noopener">Amazon ships around 1.6 million packages per day</a>, and it is therefore most desirable to have a modern, accurate system for tracking inventory. One possible component of such a system could be to monitor the number of items in each package being processed. Images of packages are captured as Amazon warehouse worker robots perform Fulfillment Centre operations and these would form the input data to be analysed for inventory management. 

## Problem statement
This project uses AWS Sagemaker to train an ML model on a subset of the ABID to count the number of objects in each bin. Although in reality we would want to optimise accuracy as much as possible in line with business priorities, in our case the focus was on building an ML pipeline that utilises best practices and could in principle be deployed to production. That is, we assume that necessary initial work regarding thorough EDA, data quality and prototyping considerations have already been carried out and we are concerned with taking a model, training it (including a hyperparameter optimisation step) on Sagemaker, and preparing and testing it for inference rather than optimising the performance of the model itself. 

## Explanation of pipeline, notebooks and scripts
The project consists of the following steps.
1. Download and arrange data.
2. Train a model on Sagemaker and conduct hyperparameter optimisation.
3. Train model on the full dataset with best hyperparameters, including debugging and profiling on Sagemaker.
4. Deploy trained model to an endpoint and confirm that we can make an inference successfully.
5. Use the model to demonstrate a batch transform on Sagemaker. 

All code to run the project is contained in two notebooks – `data preparation and eda.ipynb`, `sagemaker.ipynb` – and five Python scripts – `download_and_arrange_data.py`, `normalize_local.py`, `train.py`, `hpo.py` and `inference.py`. Step 1 is covered in `data preparation and eda.ipynb`, and the code and instructions for steps 2–5 are given in `sagemaker.ipynb`. For a general discussion of methodology and results, see `report.pdf`.

## Project setup and installation
An AWS SageMaker instance ml.t3.medium type and the following software frameworks were used for model training and analysis:

- Python 3.6
- Pytorch: 1.8

To re-create this project, create an ml.t3.medium notebook in Sagemaker, upload `sagemaker.ipynb` and the `scripts` folder. From the terminal run `download_and_arrange_data.py` and then run every cell in the notebook. Note that to carry out the batch transform, test images to be processed should be moved to a local folder on Sagemaker.


## Dataset
The full ABID of \~500,000 images can be found <a ref="https://registry.opendata.aws/amazon-bin-imagery/" target="_blank" rel="noopener">here</a>, but we use a subset provided by Udacity. For more details on the data see `data preparation and eda.ipynb`.

### Next steps
- Migrate code from `sagemaker.ipynb` to a stand-alone python script.
- Refactor code to include type hints and function documentation.
- Move code to output graphs to a separate python file.
- Add analysis of transforms.