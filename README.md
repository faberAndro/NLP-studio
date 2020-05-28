# NLP-studio


------ WHAT "ACTIONS_EXTRACTION" DOES -------

This program takes as input a text file, assumed to be a transcript from a meeting or conversation.
It tokenizes the text in sentences, then search through each sentence some 'attention words' or 'chuncks'.
The set of attention chuncks is stored in the 'attention_chuncks_vocab.txt' file (same folder).
The program then outputs the input text highlighting each sentences where at least one attention word/chunck has been found.
More attention words have been found in the sentence, stronger is the highlighting (up to 5 degrees of colors).
Highlighted sentences comes numbered within the printed text.   
Finally, a simple list of the highlighted sentences is printed out, numbered accordingly. 
Note: the program reduces all to lowercase.


------ HOW TO RUN (please read) -------

"Actions_Extraction_001.py" is the main file to run.
Please check the file "attention_chuncks_vocab.txt" is copied in the same folder.
This program takes as default the input "sample_001.txt".
N.4 additional sample files are provided in this repository for functionality test. These have been taken by BBC news. Some of them have been amended by insertion of some dedicated sentences filled with attention words. (easy to recognize as in capital letters - in source files) 
The program provides assisted input by keyboard.
To load the other 4 examples or other texta at your choice, simply write at prompt the filename you wish - completed with extension. 
Files saved in the program folder do not need their path to be specified. 

