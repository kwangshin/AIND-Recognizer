import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        min_bic_score = float('inf')
        best_hmm_model = None

        for current_n_components in range(self.min_n_components, self.max_n_components + 1):
            # Tip: The hmmlearn library may not be able to train or score all models. Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:
                # Get hmm model of current component.
                hmm_model = self.base_model(current_n_components)
                # Get the log likelihood.
                log_likelihood = hmm_model.score(self.X, self.lengths)

                # Ge the number of data points
                number_of_data_points = len(self.lengths)

                # Compute the Bayesian information criteria: BIC = -2 * logL + p * logN
                # Reference : https://en.wikipedia.org/wiki/Bayesian_information_criterion
                #             http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
                #             https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                # * logL = log likelihood.
                # * N = the number of data points.
                #
                # ------ From the review -----------------------
                # Here is some more complete info on p as it pertains here:
                # "Free parameters" are parameters that are learned by the model (as opposed to ones that are provided to it, like features). We are working on some content to clarify this,because as you point out - how will they know exactly. It is complicated by the fact that the number is different, depending on how you set up your model.
                # In this project, however, we are using the "diag" in the hmmlearn model and we are not specifying starting probabilities. Therefore, if we say that m = num_components and f = num_features, the free parameters are a sum of:
                # (1) The free transition probability parameters, which is the size of the transmat matrix less one row because they add up to 1 and therefore the final row is deterministic, so m*(m-1)
                # (2) The free starting probabilities, which is the size of startprob minus 1 because it adds to 1.0 and last one can be calculated so m-1
                # (3) Number of means, which is m*f
                # (4) Number of covariances which is the size of the covars matrix, which for "diag" is m*f
                # * p = (1) + (2) + (3) + (4)
                #     = m*(m-1) + (m-1) + m*f + m*f
                #     = m^2 - m + m - 1 + m*f + m*f
                #     = m^2 + 2*m*f - 1
                # ---------------------------------------------------------------------
                # * p = p denotes the number of independent parameters of the model..
                #   p = m^2 + km - 1.
                #     1) m : number of states in the Markov chain of the model.)
                #     2) k : single numeric value representing the number of parameters of the underlying distribution of the observation process (e.g. k=2 for the normal distribution (mean and standard deviation)).
                 # ---------------------------------------------------------------------

                number_of_parameters = math.pow(current_n_components, 2) + 2*current_n_components*(number_of_data_points) - 1

                bic_score = -2 * log_likelihood + number_of_parameters * math.log(number_of_data_points)
                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_hmm_model = hmm_model
            except:
                pass

        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i))) - 1/(M-1)SUM(log(P(X(all but i))))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        max_dic_score = float('-inf')
        best_hmm_model = None

        for current_n_components in range(self.min_n_components, self.max_n_components + 1):
            # Tip: The hmmlearn library may not be able to train or score all models. Implement try/except contructs as necessary to eliminate non-viable models from consideration.
            try:
                hmm_model = self.base_model(current_n_components)
                current_log_likelihood = hmm_model.score(self.X, self.lengths)
                others_log_likelihood_list = []
                for current_word in self.hwords:
                    # We need to calculate except current word.
                    if current_word != self.this_word:
                        current_X, current_lengths = self.hwords[current_word]
                        current_log_likeliyhood = hmm_model.score(current_X, current_lengths)
                        others_log_likelihood_list.append(current_log_likeliyhood)
                # M will be the the number of lielihood to be summed.
                value_of_m = len(others_log_likelihood_list)
                # Get DIC score.
                # DIC = log(P(X(i))) - 1/(M-1)SUM(log(P(X(all but i))))
                dic_score = current_log_likelihood - 1/value_of_m*sum(others_log_likelihood_list)

                if dic_score > max_dic_score:
                    max_dic_score = dic_score
                    best_hmm_model = hmm_model
            except:
                pass

        return best_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # If there is only 1 sequence, 
        #     then let's select the model with value self.n_constant.
        if len(self.sequences) < 2:
            return self.base_model(self.n_constant)

        # The default number of splits is 3.
        number_of_splits = 3

        if len(self.sequences) is 2:
            # If the number of sequences is 2,
            #    then use 2 as the number of splits.
            # Otherwiese, keep the 3 as the number of splits.
            number_of_splits = 2

        max_avg_log_likelihood = float('-inf')
        best_num_components = None

        for cv_n_components in range(self.min_n_components, self.max_n_components + 1):
            # List of log likelihood of this component.
            log_likelihood = []

            split_method = KFold(number_of_splits)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # Tip: The hmmlearn library may not be able to train or score all models. Implement try/except contructs as necessary to eliminate non-viable models from consideration.
                try:
                    # Combine sequence of training data.
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    # Get the model for training data.
                    model = GaussianHMM(n_components=cv_n_components, covariance_type="diag", 
                                    n_iter=1000, random_state=self.random_state, 
                                    verbose=False).fit(train_X, train_lengths)
                    # Combine sequence of test data.
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    # Get the log likelihood of test data using the model.
                    cv_log_likelihood = model.score(test_X, test_lengths)
                    # Add current log likelihood into the list.
                    log_likelihood.append(cv_log_likelihood)
                except:
                    # Catching all exceptions and just continue the loop.
                    pass

            # Get the average log likelyhood.
            cv_avg_log_likelihood = np.average(log_likelihood)
            # If this average log likelyhood is the max value, then store for result.
            if  cv_avg_log_likelihood > max_avg_log_likelihood:
                max_avg_log_likelihood = cv_avg_log_likelihood
                best_num_components = cv_n_components

        return self.base_model(best_num_components)
