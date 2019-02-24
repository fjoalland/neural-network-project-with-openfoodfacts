# Neural network with tenserflow
In pure Python and using tenserflow.
This project aims to train on a data set to learn if food is good or not. A faith that he is well trained, he will evaluate food to determine their quality.
## The  script require
- Pandas
- Matplotlib
- Tensorflow (be sure to have a version =< 3.6, Tensorflow doesn't support 3.7)

## Add new Data to test
If you want to add new ingredients to evaluate, you must add them to the CSV file repas_touristes.csv
![alt text](https://zupimages.net/up/19/08/rihf.png)

## Usage
If you want to run the comparisons yourself, just go

    python .\neuronal.py
    
## Run
We start to train our neural network with a dataset, this dataset comes from openfoodfacts. The graph below represents:
- in red: unhealthy foods
- in green: alliments good for health

![](https://image.noelshack.com/fichiers/2019/08/7/1551024390-capture.png)

Once trained, we display the elements it will have to evaluate. The color of the curves are blue because it hasn't yet evaluated whether or not the food is good or not for health.

![](https://image.noelshack.com/fichiers/2019/08/7/1551024555-capture.png)

Showing prediction of our neural network, we can notice that it deduces that a hamburger is bad for health while organic milk is good for health.

![](https://image.noelshack.com/fichiers/2019/08/7/1551024812-capture.png)

Display of the component graph of our tested ingredients:
- in red: unhealthy foods
- in green: alliments good for health

![](https://image.noelshack.com/fichiers/2019/08/7/1551024944-capture.png)

## Dependencies
- Python3

[alt_text](https://d1q6f0aelx0por.cloudfront.net/product-logos/6bd224a8-e827-4593-b5b4-483338e9999e-python.png)
![alt_text](https://cdn-images-1.medium.com/max/480/1*cKG1LJvVTaWqSkYSyVqtsQ.png)

