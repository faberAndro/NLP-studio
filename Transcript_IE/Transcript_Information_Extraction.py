import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import time
import requests
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_GloVe_dictionary():
    """ Load GloVe vectors from file 'glove.6B.50d.txt' if in the local directory.
    Alternatively, load the vector from "https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt"

    Returns: a dictionary of 400.000 50-dim GloVe Vectors

    """
    def default_value():
        return np.ones(50)
    word_embeddings_vectors = defaultdict(default_value)
    web = False
    try:
        f = open('glove.6B.50d.txt', 'rt', encoding='utf-8')
        print('loading GloVe word vectors from file ... pelase wait a minute ...')
    except OSError:
        web = True
        print('local GloVe file not available\nLoading from web resource https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt')
        print('... DOWNLOADING FILE ...\nPlease wait: this may take some minutes...')
        web_req = requests.get("https://archive.org/download/glove.6B.50d-300d/glove.6B.50d.txt")
        print('DOWNLOAD FINISHED!')
        f = web_req.text.split('\n')
        f.pop(-1)
    print('Loading word vectors to memory...\nplease wait...')
    for line in f:
        values = line.split()
        keyword = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings_vectors[keyword] = coefs
    print('... done!')
    if not web:
        f.close()
    return word_embeddings_vectors


def copy_structured_text(structured_text):
    """

    Args:
        structured_text: list of dictionaries

    Returns: a copy of the input by value

    """
    one_copy = []
    for x in structured_text:
        one_copy.append(x.copy())
    return one_copy


def _0_pre_processing(file_name):
    """

    Args:
        file_name: text file containing the transcript

    Returns: N.2 list of dictionaries.
                pre_processed_text:
                    Keys are: character' name, type [speech or description], text
                sequential_sequences:
                    same list but with texts tokenized and an additional key 'index' containing a list of sentence indexes
    """
    pre_processed_text = []
    character_speaking = False
    describing_scene = False
    character_speech = ''
    scene_description = ''
    chunk_dictionary = {}
    n_spaces1 = '\t'.expandtabs(25)
    n_spaces2 = '\t'.expandtabs(10)
    try:
        with open(file_name, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                if character_speaking:
                    if line[0:10] == n_spaces2:
                        character_speech += line.replace('\n', ' ').lstrip()
                    elif line == '\n':
                        chunk_dictionary['text'] = character_speech
                        character_speaking = False
                        pre_processed_text.append(chunk_dictionary)
                elif line[0:25] == n_spaces1:
                    character_speaking = True
                    character_speech = ''
                    name = line.replace('\n', ' ').lstrip()
                    chunk_dictionary = {'type': 'speech', 'name': name.strip()}
                elif describing_scene:
                    if line[0] != '\n':
                        scene_description += line.replace('\n', ' ')
                    else:
                        chunk_dictionary['text'] = scene_description
                        describing_scene = False
                        pre_processed_text.append(chunk_dictionary)
                elif line[0] not in ['\n', ' ']:
                    describing_scene = True
                    scene_description = line.replace('\n', ' ')
                    chunk_dictionary = {'type': 'description', 'name': None}
        file.close()
    except OSError:
        print("please place a file named \"transcript.txt\" into the program folder, then retry")
        exit(0)

    sequential_sentences = copy_structured_text(pre_processed_text)
    index = 0
    for c, chunk in enumerate(pre_processed_text):
        s_list = nltk.sent_tokenize(chunk['text'])
        sequential_sentences[c]['text'] = s_list
        n_sentences = len(s_list)
        indexes = np.arange(n_sentences) + index
        if n_sentences:
            sequential_sentences[c]['indexes'] = indexes
        else:
            sequential_sentences[c]['indexes'] = None
        index += n_sentences

    return pre_processed_text, sequential_sentences


def _1_extract_normalized_keywords(text):
    """

    Args:
        text: the transcript text as a nested list

    Returns: a set of normalized keywords

    """
    return _3_analyse_word_frequencies(text, graphs=False)


def _2_remove_stopWords_and_punctuation(pre_processed_text, way):
    """

    Args:
        pre_processed_text: transcript in form of list of dictionaries
        way: string that could be 'nltk_way' or 'regex_way'. 'regex_way' is a draft. Use the other one.

    Returns: text
                - lowered case
                - tokenized in sentences and words
                - without punctuation
                - without stop words

             text_without_stop_words_and_punctuation: same form as the input
             purged_text_as_lists: as the input, but text is now a nested list of words in sentences
             text_for_summary: same as above, but where sentences have been not tokenized into words

    """
    text_without_stop_words_and_punctuation = copy_structured_text(pre_processed_text)
    purged_text_as_lists = copy_structured_text(pre_processed_text)
    text_for_summary = copy_structured_text(pre_processed_text)
    stop_words_list = set(stopwords.words('english'))   # remove any possible duplicate from stopwords corpus

    if way=='regex_way':
        stop_words_regex_list = list(map((lambda stop_w: ''.join([r'(\W)', stop_w, r'(\W)'])), stop_words_list))
        text_without_stop_words = pre_processed_text.lower()
        for stop_word in stop_words_regex_list:
            text_without_stop_words = re.sub(stop_word, r'\1', text_without_stop_words)

    if way=='nltk_way':
        for s, chunk in enumerate(text_without_stop_words_and_punctuation):
            lowered_sentences = chunk['text'].lower()
            tokenized_sentences = nltk.sent_tokenize(lowered_sentences)

            sequential_words_in_chunk = []
            list_of_sentences_in_words = []
            list_of_complete_sentences = []
            for tokenized_sentence in tokenized_sentences:
                sentence_no_punkt = re.sub('\\W', ' ', tokenized_sentence)
                sentence_in_words = nltk.word_tokenize(sentence_no_punkt)
                list_of_complete_sentences.append(sentence_in_words.copy())
                sentence_in_words = [word for word in sentence_in_words if word not in stop_words_list]
                list_of_sentences_in_words.append(sentence_in_words)
                sequential_words_in_chunk.extend(sentence_in_words)

            purged_text_as_lists[s]['text'] = list_of_sentences_in_words
            text_without_stop_words_and_punctuation[s]['text'] = ' '.join(sequential_words_in_chunk)
            text_for_summary[s]['text'] = list_of_complete_sentences

    return text_without_stop_words_and_punctuation, purged_text_as_lists, text_for_summary


def _3_analyse_word_frequencies(text_as_nested_lists, graphs=True):
    """

    Args:
        text_as_nested_lists: transcript in form of nested lists
        graphs: boolean to control if the graphs need to be shown.

    stemming and/or lemmatization are applied

    Returns: N.2 graphs of word frequencies if graphs is True, together with a table of frequencies
             a set of normalized keywords if graph is false

    """
    full_word_list = []
    for chunk in text_as_nested_lists:
        for sentences in chunk['text']:
            full_word_list.extend(sentences)

    # <editor-fold desc="STEMMING AND LEMMATIZATION">
    ps = PorterStemmer()
    lem = WordNetLemmatizer()
    full_word_list = [lem.lemmatize(ps.stem(x)) for x in full_word_list]
    result_lemmatized = set([lem.lemmatize(x) for x in full_word_list])
    result_stemmed = set([ps.stem(x) for x in full_word_list])
    result_stemmed_plus_lemmatized = set([lem.lemmatize(ps.stem(x)) for x in full_word_list])

    # </editor-fold>

    if graphs:
        frequencies = nltk.FreqDist(full_word_list)
        frequencies.tabulate()
        frequencies.plot(50, cumulative=False, title='Frequencies in structured text')

    full_word_list.extend([chunk['name'].lower() for chunk in text_as_nested_lists if chunk['name'] is not None])
    if graphs:
        frequencies = nltk.FreqDist(full_word_list)
        frequencies.plot(50, cumulative=False, title='Frequencies in structured text with character names')
        result = frequencies
    else:
        print('KEYWORDS')
        # print('lemmatized:\n', result_lemmatized, len(result_lemmatized))
        # print('stemmed:\n', result_stemmed, len(result_stemmed))
        print('stemmed then lemmatized:\n', result_stemmed_plus_lemmatized, len(result_stemmed_plus_lemmatized))
        # print(result_lemmatized - result_stemmed, result_stemmed - result_lemmatized)
        result = result_stemmed
    return result


def _4_extract_names(text, structured_text):
    """

    Args:
        text: transcript in unstructured shape
        structured_text: transcript as nested lists. Only tokenized sentences
    Returns: candidate proper names' set

    """
    text = re.sub('\\W', ' ', text)
    stop_words = ''# set(stopwords.words('english'))
    text_splitted = text.split()
    text_no_stop_words = [x for x in text_splitted if x not in stop_words]
    tagged_sentences = nltk.pos_tag(text_no_stop_words)
    # tagged_lowercase_sentences = nltk.pos_tag((text.lower().split()))
    # grammar = "NAME-- : {<NNP>}"
    # parser = nltk.RegexpParser(grammar)
    # list_of_names = parser.parse(tagged_sentences)
    list_of_names = [(i, x[0]) for i, x in enumerate(tagged_sentences) if x[1]=='NNP']
    # names_position = [n[0] for n in list_of_names]
    # comparison_list = [(i, y) for i, y in enumerate(tagged_lowercase_sentences) if i in names_position]
    # print(list_of_names, '\n\n', comparison_list)
    set_of_names = sorted(set((x[1].lower() for x in list_of_names)))
    # extracted_chunks = parser.parse(tagged_sentence)
    # baseline_tagger = nltk.tag.UnigramTagger(model=...)

    for chunk in structured_text:
        for sentence in chunk['text']:
            tag_list = nltk.pos_tag(sentence)
            print(tag_list)

    return tagged_sentences, set_of_names


def _5_summarise(word_embedding_vectors, text_of_sentences, cleaned_text_as_lists):
    """

    Args:
        word_embedding_vectors: GloVe vectors dictionary
        text_of_sentences: "sequential sentences" as per second output of pre-processing _0_ function
        cleaned_text_as_lists: normalized text as output of function #2, but without stemming and/or lemmatization

    TextRank algorithm

    Returns: summary of the transcript as the first 20% ranked sentences

    """
    # Assign word vectors to transcript words - sequentially
    sentence_vectors = []
    for sent_chunk in cleaned_text_as_lists:
        for sentence in sent_chunk['text']:   # 'sentence' is a list of splitted words
            if len(sentence)>0:
                sentence_vector = []
                print(sentence)
                for word in sentence:
                    sentence_vector.append(word_embedding_vectors[word])
                sentence_vector = np.average(np.asarray(sentence_vector), axis=0)
                assert(sentence_vector.shape == (50,)), print(sentence, sentence_vector)

                sentence_vectors.append(sentence_vector)

    n = len(sentence_vectors)

    # build up similarity matrix
    similarity_matrix = np.zeros((n, n))
    remaining_time = '?'
    for i in range(n):
        t1 = time.time()
        print(round(i/n, 3)*100, '%', 'remaining time [sec]:', remaining_time)
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 50), sentence_vectors[j].reshape(1, 50))[0, 0]
        t2 = time.time()
        remaining_time = round((t2 - t1), 0) * (n-i)
    # apply text_rank / page_rank algorithm
    print('senteces: ', n, ' - building now the graph... ')
    nx_graph = nx.from_numpy_array(similarity_matrix)
    nx.draw(nx_graph)
    input()
    print('calculating the pagerank scores')
    scores = nx.pagerank(nx_graph, max_iter=200)
    ranked_sentences = sorted(scores.items(), key=lambda score: score[1], reverse=True)

    # extract top ranked N sentences from the original text corpus
    n_top = 12
    for s in ranked_sentences[:n_top]:
        for chunk in text_of_sentences:
            if s[0] in chunk['indexes']:
                position = np.where(chunk['indexes'] == s[0])[0][0]
                top_sentence = chunk['text'][position]
                if chunk['name']:
                    output_sentence = chunk['name'] + ' says: \"' + top_sentence + '\"'
                else:
                    output_sentence = top_sentence
                print(output_sentence)


def _6_additional_text_attribution(word_embedding_vectors, additional_text_file, cleaned_text):
    """

    Args:
        word_embedding_vectors: Glove vectors dictionary
        additional_text_file: file name of additional speech to assign
        cleaned_text: structured normalized text in shape of nested lists, as per output of function 2

    Returns: similarity coefficients and candidate speaker to be assigned to the input text

    """
    # LOAD AND NORMALIZE THE ADDITIONAL TEXT
    af = open(additional_text_file, 'rt', encoding='utf-8')
    add_text = af.read().lower()
    add_text_sentences = nltk.sent_tokenize(add_text)
    add_text_no_punkt = [re.sub('\\W', ' ', x) for x in add_text_sentences]
    add_text_in_splitted_words = []
    for x in add_text_no_punkt:
        add_text_in_splitted_words.extend(nltk.word_tokenize(x))
    stop_words = set(stopwords.words('english'))
    add_text_purged = [x for x in add_text_in_splitted_words if x not in stop_words]

    # assign a speech vector to the additional text
    add_text_speech_vector = np.average(np.asarray([word_embedding_vectors[word] for word in add_text_purged if word_embedding_vectors[word]]), axis=0)

    # Assign "speech vectors" to each character's speech in a dictionary
    characters_vectors = {}     # output initialization
    for s in cleaned_text:
        if s['name']:
            characters_vectors[s['name']] = []

    sentence_vectors = []
    for sent_chunk in cleaned_text:
        if sent_chunk['name'] is not None:
            for sentence in sent_chunk['text']:   # 'sentence' here is the full piece of speech
                if len(sentence)>0:
                    sentence_vector = np.average(np.asarray([word_embedding_vectors[word] for word in sentence if word_embedding_vectors[word]]), axis=0)
                    sentence_vectors.append(sentence_vector)
                speech_average_vector = np.average(sentence_vectors, axis=0)
                characters_vectors[sent_chunk['name']].append(speech_average_vector)
    # for person in characters_vectors:
    #     characters_vectors[person] = np.average(characters_vectors[person], axis=0)

    # comparison doing cosine similarities: if the overall similarity is less than 0.5, it outputs "not enough similarities found"
    treshold = 0.5
    selected_character = 'no one'
    sim_grade = 0
    for person in characters_vectors:
        average = 0
        for i, single_speech in enumerate(characters_vectors[person]):
            cs = cosine_similarity(add_text_speech_vector.reshape(1, 50), single_speech.reshape(1, 50))[0, 0]
            print(person, 'speech %s similarity coef:' %str(i), cs)
            average += cs
        average = average/len(characters_vectors[person])
        if average > sim_grade:
            sim_grade = average
            selected_character = person
    if sim_grade > treshold:
        print('this text is attributed to:', selected_character)
    else:
        print('Not enough similarities found.')


if __name__ == '__main__':
    # TODO: check if the key exists in embedding words when using generators

    # LOAD TRANSCRIPT AS PLAIN UNSTRUCTURED TEXT FROM FILE to "unstructured_text"
    try:
        with open("transcript.txt") as transcript_file:
            unstructured_text = transcript_file.read()
        transcript_file.close()
    except OSError:
        print("please place a file named \"transcript.txt\" into the program folder, then retry")
        exit(0)

    # ASK USER FOR CHOICES
    word_vectors_are_loaded = False
    go_on = 'y'
    while go_on == 'y':
        choice = None
        message = '''
        Please input the number of function to run:
        1: set of normalized keywords (lowered, purged from stop words and punctuation)
        (2): not used
        3: graphs and table of word frequencies (stemmed and/or lemmatized)
        4: set of (candidate) proper names found in text
        5: Summary attempt
        6: additional text attribution to characters
        '''
        while choice not in ['1', '3', '4', '5', '6']:
            choice = input(message)

        start_time = time.time()
        # PERFORMS OPERATIONS ON TRASCRIPT
        text_first_processed, text_in_sentences = _0_pre_processing("transcript.txt")
        purged_text, purged_text_in_lists, text_to_go_to_5 = _2_remove_stopWords_and_punctuation(text_first_processed, 'nltk_way')
        if choice == '1':
            _1_extract_normalized_keywords(purged_text_in_lists)
        if choice == '3':
            freq_list = _3_analyse_word_frequencies(purged_text_in_lists)
        if choice == '4':
            test_list, proper_names = _4_extract_names(unstructured_text, purged_text_in_lists)
            print(proper_names)
        if choice in ['5', '6']:
            if not word_vectors_are_loaded:
                word_embeddings = load_GloVe_dictionary()
                word_vectors_are_loaded = True
            if choice == '5':
                print('------ Summary performed EXCLUDING stop words: ------ ')
                _5_summarise(word_embeddings, text_in_sentences, purged_text_in_lists)
                print('\n------- Summary performed using ALL words: ------- ')
                _5_summarise(word_embeddings, text_in_sentences, text_to_go_to_5)
            if choice == '6':
                _6_additional_text_attribution(word_embeddings, "unknown_speech.txt", purged_text_in_lists)

        end_time = time.time()
        print('(Running time: ', end_time - start_time, 'sec.)')

        # WRITE NEW TEXT TO FILE
        if False:
            with open('amended_transcript.txt', 'wt', encoding='UTF-8') as amended_transcript_file:
                amended_transcript_file.write(amended_transcript_text)
            amended_transcript_file.close()

        go_on = input('Do you want to try other functions ? (y/n)  ')

