import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    # Get for entire db of items as (X, lengths) tuple for use with hmmlearn library
    test_set_all_items = test_set.get_all_Xlengths()
    # Get a copy of the dictionary’s list of (key, value) pairs.
    all_items_list_of_pairs = test_set_all_items.items()
    # Loop all items.
    for index, (X, lengths) in all_items_list_of_pairs:
        word_likeliyhood_dictionary = dict()

        best_guess_word = None
        max_log_likelihood = float('-inf')

        for trained_word, trained_model in models.items():
            # Tip: The hmmlearn library may not be able to train or score all models. Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:
                # Get the log likelihood.
                current_log_likelihood = trained_model.score(X, lengths)
                # Add the dictionary of (word: log likelihood) pair.
                word_likeliyhood_dictionary[trained_word] = current_log_likelihood

                if current_log_likelihood > max_log_likelihood:
                    # If the current likelihood is max,
                    #     then consider it as the best guess word!
                    max_log_likelihood = current_log_likelihood
                    best_guess_word = trained_word
            except:
                # If current item cannt be trained, then set minimum number.
                word_likeliyhood_dictionary[trained_word] = float('-inf')
        # Add current probability which is a dictionary type.
        probabilities.append(word_likeliyhood_dictionary)

        guesses.append(best_guess_word)

    return probabilities, guesses
