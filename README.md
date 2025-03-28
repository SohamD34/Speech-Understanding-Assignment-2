# Speech-Understanding-Assignment-2
## Submission by - Soham Deshmukh (B21EE067)
Code repository for submission to the Assignment 2 of CSL7770: Speech Understanding Course

In order to run the code files provided, clone this repository into your local machine.
```
> cd <location>
> git clone https://www.github.com/SohamD34/Speech-Understanding-Assignment-2.git
> cd Speech-Understanding-Assignment-2/
```
Create a virtual Python environment.
```
> python3 -m venv b21ee067
```
Activate the environment and install the necessary Python libraries and packages.
```
> source b21ee067/bin/activate
> pip install -r requirements.txt
```


## Question 1
The task is to enhance the speech of speakers in a multi-speaker environment.
For this, download the VoxCeleb1 and VoxCeleb2 datasets from our data source using the ```data_ingestion.py``` script in the ```scripts``` folder.
```
> cd scripts/
> python data_ingestion.py
```
You should be able to see a ```data``` folder with ```wav``` and ```aac``` folders for VoxCeleb1 and VoxCeleb2 datasets respectively.

