# NLP-studio


------ WHAT "ACTIONS_EXTRACTION" DOES -------

This program takes as input a text file assumed to be a transcript from a meeting or conversation.
It tokenizes the text in sentences, then searches through each sentence several 'attention words' or 'chunks'.
The attention chuncks set is stored in the 'attention_chuncks_vocab.txt' file (same folder).
The program outputs the same input text but highlighting each sentences, where at least one attention word/chunk has been found.
More attention words are found in one sentence, stronger the highlight is (up to 5 degrees of colors).
Highlighted sentences come numbered within the printed text.   
Finally, a simple list of these sentences is printed out, cross-referenced to the corpus. 
Note: the program reduces all to lowercase.


------ HOW TO RUN (please read) -------

"Actions_Extraction_001.py" is the main file to run.
It runs and works by command line (no GUI).
Please check the file "attention_chuncks_vocab.txt" is copied into the same folder.
This program takes as default input file "sample_001.txt".
N.4 additional sample files have been provided in this repository for functionality test. They have been downloaded by BBC webpages. Some of them have been amended by insertion of dedicated sentences filled with attention words (additions are easy to recognize as in capital letters through source files) 
To load the other 4 examples or other text at your choice, simply write at prompt the filename you like - completed with extension. 
Files saved into the program folder do not need their path to be specified at prompt. 

