<<<<<< Weather Prediction >>>>>>>
________________________________________________________________________________________________________
#Technologies
Project is created with Python 3.6
________________________________________________________________________________________________________
#Description
It predicts the average, minimum or maximum temperature given a date.
It plots a graph between temperature values for the specified time duration.
________________________________________________________________________________________________________
#Setup

Use pip for python2 . This project runs on python3. SO it is pip3. Depends on your system config.

pip3 install matplotlib
pip3 install numpy
pip3 install scikit_learn
pip3 install scipy
pip3 install pandas
sudo apt-get install python3-tk
________________________________________________________________________________________________________
#Files

#Regression Algorithms
BayesianRidge.py
decision tree reg.py
KNeighbours.py
linear.py
RandomForest.py
SDG.py
SVR.py
-------- All these files run the said algorithm. 

gui.py - Python GUI file that consolidates everything
Comparision.xlsx - Mean sqaure error values for different regression algorithms
MTV1.xlsx - Average, Minimum and Maximum values of months obtained through data.gov.in - used in generation of dataset
generate.py - Dataset generation algorithm
WeatherDATA1.csv - Dataset
________________________________________________________________________________________________________
#How to run ?
To run a file, "python3 <nameoffile.py>"
________________________________________________________________________________________________________
#How to view varying results ? 
1) Open the required algorithm file
2) Bottom of the screen has the date required to predict. eg: [27,11,2019]
3) Edit it to a summer date eg : [27,05,2019] and then run=== in summer average will be higher , say 30 or more
4) Edit it to a winter date eg : [02,12,2019] and then run=== in winter average temp be lower , say 27 or less.

Format for Weather Statistics: DD-MM-YYYY
Format for Rainfall Statistics: MM-YYYY
________________________________________________________________________________________________________
#BEST RESULTS : 
Best results are obtained for KNeighbours,Decision Trees,Randomforest.
Others have poor results. THis is the conlusion of our project.

 
