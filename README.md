# NLP-studio


------ WHAT "ACTIONS_EXTRACTION" DOES -------

This program takes as input a text file assumed to be a transcript from a meeting or conversation.
It tokenizes the text in sentences, then searches through each sentence several 'attention words' or 'chunks'.
The attention chuncks set is stored in the 'attention_chunks_vocab.txt' file (same folder).
The program outputs the same input text but highlighting each sentences, where at least one attention word/chunk has been found.
More attention words are found in one sentence, stronger the highlight is (up to 5 degrees of colors).
Highlighted sentences come numbered within the printed text.   
Finally, a simple list of these sentences is printed out, cross-referenced to the corpus. 
Note: the program reduces all to lowercase.


------ HOW TO RUN -------

"Actions_Extraction_001.py" is the main file to run.
It runs and works by command line (no GUI).
PLEASE DO ALL THE FOLLOWING STEPS:
1. Download the full zip folder and do not change or move the file in it. 
2. Ensure to have installed the latest python interpreter (or at least 3.7)
3. Ensure especially the files "attention_chunks_vocab.txt" and "sample_001.txt" are copied into the folder.
4. Install the following python modules (if not already on your machine):     
    * nltk      (run from terminal "pip install nltk") 
    * colored   (run from terminal "pip install colored")
    * py-getch  (run from terminal "pip install py-getch")
5. Install the following 'punkt' dependency for ntlk -> from the python interpreter command line run first "import nltk" and then 
"nltk.download('punkt')"  
6. Double click on Actions_Extraction_001.py or run the program from your system terminal
7. If you still miss some dependencies from nltk, try run the command "nltk.downloader()" from the interpreter. When in the pop up window, download the "popular" packages from "collection" tag.

This program takes as default input file "sample_001.txt".
N.4 additional sample files have been provided in this repository for functionality test. They have been downloaded by BBC webpages. Some of them have been amended by insertion of dedicated sentences filled with attention words (additions are easy to recognize as in capital letters through source files) 
To load the other 4 examples or other text at your choice, simply write at prompt the filename you like - completed with extension. 
Files saved into the program folder do not need their path to be specified at prompt. 


----- BUGS FIXED -----

- the attention word file is now checked on loading, to strip out newline or blank elements.  
- Spelling for chunks corrected.
