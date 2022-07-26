
### Table of Contents

1. [Installation](#installation)
2. [Project Summary](#summary)
3. [File Descriptions](#files)
4. [Instructions](#instructions)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Project Summary<a name="summary"></a>

A web app that classifies disaster messages into categories so that the right help will reach the people in need. An emergency worker can input a disaster message and get classification results in several categories. The web app also displays visualizations of the training data used.

## File Descriptions <a name="files"></a>

There are three folders - app, data and models.

The app folder contains run.py which runs the web app and a templates folder that contains the html files - go.html and master.html.

The data folder contains the raw csv files - disaster_categories.csv and disaster_messages.csv, and the process_data.py file, which loads, cleans and save the cleaned data to a sqlite database.

The models folder contains train_classifier.py, which builds, trains, evaluate and save the trained model as a pickle file.

## Instructions<a name="instructions"></a>

First, run the ETL pipeline to clean and store the data in the sqlite database via the command "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db".

Next, run the ML pipeline that trains the classifier and saves it as a pickle file via the command "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl".

Run the web app via the command "python run.py".
