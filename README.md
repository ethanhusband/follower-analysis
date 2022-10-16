# File structure and use of each file

In the directory src, a folder is created for each section of the dataset that is to be analysed. 
There is also a data folder containing the original dataset.
All other files included in src form the data processing pipeline.

# Instructions on how to run the code.

In a terminal, first navigate to the folder you wish to generate the analysis for.
For example, if you wish to generate analysis for Coherence, navigate to src/CoherenceData.
From this, run 
`python OutputData.py`
and you will see all the analysed data for that section of the dataset appear in the folder.
Most of the data will appear in output.txt, as well as regression graphs as png files.

