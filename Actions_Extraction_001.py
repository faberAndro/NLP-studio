try:
    import nltk
    from colored import fg, bg, attr
    from getch import pause
except ModuleNotFoundError:
    print('''For this program to work, the following python modules need to be installed (use this: pip install 'module_name'):
    * nltk
    * colored
    * py-getch  
    Please check all modules have been correctly installed and try again.
          ''')
    exit(0)
    """
    This program takes as input a text file, assumed to be a transcript from a meeting or conversation.
    It tokenizes the text in sentences, then search through each sentence some 'attention words' or 'chunks'.
    The set of attention chunks is stored in the 'attention_chunks_vocab.txt' file (same folder).
    The program then outputs the input text highlighting each sentences where at least one attention word/chunk has been found.
    More attention words have been found in the sentence, stronger is the highlighting (up to 5 degrees of colors).
    Highlighted sentences comes numbered within the printed text.   
    Finally, a simple list of the highlighted sentences is printed out, numbered accordingly. 
    """


def _from_rgb(rgb):
    """translates an rgb tuple of int to a friendly color code
    """
    return "#%02x%02x%02x" % rgb


def recognise_date_or_time():
    pass


def load_transcription():
    main_text, lowered_text, transcription_file = '', '', ''
    file_path = './'
    file_name = 'sample_001.txt'
    error_color = fg('red')
    message = '''
    Please insert the input file name (the current directory is the default one)
    or press enter to load the default file. 
    Press instead \"e\" to exit program: '''
    while True:
        default = ''
        command = input(message)
        if command=='e':
            print('Program terminated')
            exit(0)
        else:
            if command in ['\n', '']:
                transcription_file = file_path + file_name
                default = 'default file'
            else:
                transcription_file = command
            try:
                with open(transcription_file, 'rt', encoding='UTF-8') as f:
                    try:
                        main_text = f.read()
                        lowered_text = main_text.lower()
                        f.close()
                        print(' '.join(['Loaded', default, transcription_file, '\n']))
                        break
                    except UnicodeDecodeError:
                        print(error_color + 'File type not recognized. Valid files are text type (UTF-8). Please try again' + attr('reset'))
            except OSError:
                print(error_color + 'File not found or input incorrect. Please try again' + attr('reset'))

    return main_text, lowered_text


def load_attention_chunks():
    file_path = './'
    file_name = 'attention_chunks_vocab.txt'

    vocab_file = file_path + file_name
    with open(vocab_file, 'rt', encoding='UTF-8') as f:
        vocab_corpus = f.read().lower()
    f.close()
    raw_attention_words = vocab_corpus.split('\n')
    attention_words = []
    for i, a in enumerate(raw_attention_words):     # check the right syntax of attention words file
        new_a = a.replace('\t', ' ').strip()
        if a!='' and a!='\n':
            attention_words.append(new_a)
    return attention_words


def find_actions(document):
    pronouns = ['i', 'you', 'he', 'she', 'we', 'they', 'who']      # for future dev
    sentences = tuple(nltk.sent_tokenize(document))     # transcript is tokenized into sentences

    marks = []
    # !!! FOCUS POINT: THE FOLLOWING BLOCK IS THE MAIN SEARCH ALGORITHM
    for sentence in sentences:      # every sentence is assigned with the number of attention chunk occurrences in it
        n_occurrences = 0
        for attention_chunk in attention_chunks:
            if attention_chunk in sentence:
                n_occurrences += 1
        marks.append(n_occurrences)

    if any(marks):    # program proceeds only if at least one attention chunk has been found
        tagged_sentences = tuple((s, marks[i]) for i, s in enumerate(sentences))    # generated tuples: (sentence, occurrences)

        colored_output, attention_output = '', ''   # printing result
        n_highlight = 0
        color_list = [240, 220, 180, 140, 90]
        numbering_color = bg(1) + fg(255)
        for s, tagged_sentence in enumerate(tagged_sentences):
            phrase, occurrences = tagged_sentence[0], tagged_sentence[1]
            if occurrences > 0:
                n_highlight += 1
                c = color_list[min(occurrences, len(color_list))]
                color_a = _from_rgb((255, c, c))
                s_number = numbering_color + str(n_highlight) + ' ' + attr('reset')   # numbering highlighted sentences and assigning specific color
                next_sentence = s_number + bg(color_a) + fg('black') + phrase + attr('reset')
                attention_output += ' '.join([str(n_highlight), ':', fg(color_a), phrase, attr('reset'), '\n'])
            else:
                next_sentence = phrase
            colored_output = colored_output + next_sentence
        return colored_output, attention_output

    else:
        no_action_message = 'NO ACTION FOUND FROM THIS TRANSCRIPT.'
        print(bg(28), no_action_message, attr('reset'))
        return None, None


if __name__=='__main__':
    go_on = True
    while go_on:
        attention_chunks = load_attention_chunks()
        original_text, text = load_transcription()
        transcript_highlighted, list_of_actions = find_actions(text)

        if list_of_actions:
            print(transcript_highlighted + '\n')
            print(list_of_actions)

        decision = ''
        while decision not in ['y', 'n']:
            decision = input('Do you want to check another transcript? (y/n) ')
            go_on = True if decision=='y' else False
    pause('Press any key to exit ..')
