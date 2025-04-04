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

#### TASK 1
The task is to enhance the speech of speakers in a multi-speaker environment.
For this, download the VoxCeleb1 and VoxCeleb2 datasets from our data source using the ```data_ingestion.py``` script in the ```scripts``` folder.
```
> cd scripts/
> python 1.\ data_ingestion.py
```
You should be able to see a ```data``` folder with ```voxceleb1``` and ```voxceleb2``` folders for VoxCeleb1 and VoxCeleb2 datasets respectively. 
You should also see ```voxceleb1_trial_pairs.txt``` containing the training trial pairs. And the corresponding ```voxceleb_trial_pairs.csv``` for more readable and easy access during future tasks.

#### TASK 2
For this task, we first download model checkpoint from [Model link](https://drive.usercontent.google.com/download?id=1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP&export=download&authuser=0) and save it at ```scripts/Question 1/models/wavlm_base/wavlm_large_finetune.pth``` location. 
The directory structure should be like - 
```
scripts
├── Question 1
│       ├── models
│       │      └── wavlm_base
│       │            └── wavlm_large_finetune.pth
│       ├── 1. data_ingestion.py
.       ├── 2. speaker_verification.py
.           .
.           . 

```

To perform the testing of this model on VoxCeleb1 dataset, you can run the following script.
```
> python 2.\ speaker_verification.py
```
The output metrics can be observed in the logs at ```logs/question1.txt```.

Now in order to finetune this model on VoxCeleb2 dataset, we have to run the following scripts.
```
> python 3.\ WavLM_finetuning.py
```
The script will first finetune the base WavLM model using LoRA on VoxCeleb2 dataset. Then it will evaluate the finetuned model on VoxCeleb1 dataset again. The output metrics of the testing can be observed at ```logs/question1/txt```.
The metrics include AUC, accuracy, precision, recall, f1-score, EER and TAR @ 1% FAR.

#### TASK 3 & 4
Run the mixture.py first to create a data/mixtures directory containing mixture of utterances from VoxCeleb1 and VoxCeleb2.
```
> python mixture.py
```
The mixture dataset will be created at ```data/mixtures/``` folder.

Now run the sepformer script using below command to first test it directly on the mixtures dataset.
Then later, it tests the LoRA-finetuned WavLM model loaded in Sepformer architecture on the same dataset. 
```
> python 4.\ sepformer.py
```
The output accuracy metrics for both the cases can be observed at ```logs/question1.txt```.


## Question 2
The task is to extract MFCC Features from audio samples of 10 different Indian Languages and compare them.

### Task A
Download the Kaggle dataset at this URL - [Language Audio Dataset](https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages) in the location ```data/audio_dataset```.

The data directory should look like -
```
data
├── audio_dataset
│         └── archive.zip
├── voxceleb1
└── voxceleb2
```

Navigate to the source directory for Question 2.
```
> cd scripts/Question\ 2/
```
The ZIP file ```data/audio_dataset/archive.zip``` has to be unzipped using the following script.
```
> python data_ingestion.py
```
The data directory should now look like -
```
data
├── audio_dataset
│         └── Language Detection Dataset
│                        ├── Bengali
│                        ├── Gujarati
│                        ├── Hindi
│                        .
│                        .
├── voxceleb1
└── voxceleb2
```
Run the ```mel_feature_extraction.py``` script to extract the Mel Frequency Cepstral Coefficients (MFCC) and get spectrograms for each audio file of each language.
```
> python mel_feature_extraction.py
```
The MFCC features are saved at  ```data/audio_dataset/mel_features.csv```. The spectrograms are saved at ```scripts/Question 2/spectrograms```.


### Task B

To run the classifiers - SVM, RFC and KNN - run the ```classification.py``` script.
```
> python classification.py
```
All the results - best model parameters after Grid Search, training accuracy, testing accuracy, classification report - for the classification can be checked in the ```logs/question2.txt``` file.


## Note
For all the processes and subprocesses, logs are maintained at ```logs/question1.txt``` and ```logs/question2.txt```. They can be referred to understand the workflow.