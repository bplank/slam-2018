import sys
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import numpy as np
import json
from scipy.stats import entropy
import math

##### Helper functions ####

def get_char_ngrams(word):
    """ generate character 2-3 grams"""
    word = "^^{}$$".format(word) # pad
    char_ngrams = []
    for n_gram_size in [2, 3]:
        for n_gram_tuple in nltk.ngrams(word, n_gram_size):
            char_ngrams.append("_".join(n_gram_tuple))
    return char_ngrams


def get_binned_log_time(time, cumu=False, per_token=False):
    bin_end = 1 if per_token else 2
    bins = np.logspace(0, bin_end, 5, endpoint=True, base=10)
    max_bin = np.digitize(time, bins)
    if cumu:
        return (bin for bin in range(max_bin + 1))
    return max_bin


def get_binned_seen_count(seen_count, cumu=False):
    # bins are capped at 35 based on one inspected word/user distribution
    bins = 10 ** np.linspace(np.log10(1), np.log10(35),5)
    max_bin = np.digitize(seen_count, bins)
    if cumu:
        return (bin for bin in range(max_bin + 1))
    return max_bin # note that 'seen 0 times' is different from bin 0


def get_binned_days_since(today,last_seen, cumu=False):
    # bins are sensitive to short time spans; ca. in normal times: 2m, 30m, 90m, 3h, 16h, 2d, 7d
    bins = [0.001]+list(10 ** np.linspace(np.log10(.02), np.log10(7),6))
    time_passed = today - last_seen
    max_bin = np.digitize(time_passed, bins)
    if cumu:
        return (bin for bin in range(max_bin + 1))
    return max_bin


def get_binned_01_range(ratio, num_bins=10, cumu=False):
    "bin range 0-1 into num_bins"
    bins = np.linspace(0,1,num_bins)
    max_bin = np.digitize(ratio, bins)
    if cumu:
        return (bin for bin in range(max_bin + 1))
    return max_bin


def get_binned_sentence_len(ratio, num_bins=5, cumu=False):
    "bin range 0-1 into num_bins - max len en_es 14, fr_en 11"
    bins = np.linspace(0,14,num_bins)
    max_bin = np.digitize(ratio, bins)
    if cumu:
        return (bin for bin in range(max_bin + 1))
    return max_bin


def get_char_ngram_overlap_src_trg(trg_token, data_source):
    count_ngram = 0
    token_ngrams = get_char_ngrams(trg_token)
    for n_gram in token_ngrams:
        if n_gram in data_source.source_lang_char_ngram_freq_dict:
            count_ngram += 1
    if count_ngram:
        return count_ngram / len(token_ngrams)
    return 0


def get_kl_div(token, data_source):
    """KL div over character 2-3 grams between src and trg"""
    token_ngrams = get_char_ngrams(token)
    vec_trg = np.zeros(len(token_ngrams))
    vec_src = np.zeros(len(token_ngrams))
    for i, ngram in enumerate(token_ngrams):
        vec_trg[i] = data_source.target_lang_char_ngram_freq_dict.get(ngram, 0)
        vec_src[i] = data_source.source_lang_char_ngram_freq_dict.get(ngram, 0)
    if np.sum(vec_src) > 0:
        vec_trg = vec_trg / np.sum(vec_trg)
        vec_src = vec_src / np.sum(vec_src)
        ## entropy for vec_trg and vec_src
        H_trg = entropy(vec_trg)
        H_src = entropy(vec_src)
        kl = KL(vec_src, vec_trg)
        kl_norm = kl / (H_trg + H_src)
        #print(kl_norm)
        #print(get_binned_01_range(kl_norm, num_bins=500))
        return get_binned_01_range(kl_norm, num_bins=500)
    return 0


def KL(x, y):
    """ KL divergence"""
    x = np.asarray(x, dtype=np.float)
    y = np.asarray(y, dtype=np.float)
    d1 = x * np.log2(2 * x / (x + y))
    d1[np.isnan(d1)] = 0
    d = 0.5 * sum(d1)
    return d


def exists_in_dictionary(word, dict):
    """ check if word (or lower-cased word) exists in dict """
    if word in dict:
        return True
    if word.lower() in dict:
        return True
    return False

#### DataSource to load secondary data ####

class DataSource(object):
    def __init__(self,lang):

        self.target_lang,self.source_lang = lang.split("_")
        if self.target_lang == "en":
            self.target_stemmer = SnowballStemmer("english",ignore_stopwords=True)
            self.target_stopwords = set(stopwords.words('english'))
        elif self.target_lang == "es":
            self.target_stemmer = SnowballStemmer("spanish", ignore_stopwords=True)
            self.target_stopwords = set(stopwords.words('spanish'))
        elif self.target_lang == "fr":
            self.target_stemmer = SnowballStemmer("french", ignore_stopwords=True)
            self.target_stopwords = set(stopwords.words('french'))

        # load source side stemmers as well
        if self.source_lang == "en":
            self.source_stemmer = SnowballStemmer("english",ignore_stopwords=True)
        elif self.source_lang == "es":
            self.source_stemmer = SnowballStemmer("spanish", ignore_stopwords=True)

        print("Loading source (secondary) data")
        self.source_lang_word_freq_dict, self.source_lang_stem_freq_dict, self.source_lang_char_ngram_freq_dict = self.__read_raw_data_dict(
            self.source_lang, self.source_stemmer)
        self.target_lang_word_freq_dict, self.target_lang_stem_freq_dict, self.target_lang_char_ngram_freq_dict = self.__read_raw_data_dict(
            self.target_lang, self.target_stemmer)

        self.user_vocabs = self.__build_user_vocabs()

        """This dictionary of dictionaries used keys like QtTuzqbZ0303 to store global token information"""
        self.token_dict = defaultdict(dict)

        print("Loading language/country information")
        self.lang_spoken_in = defaultdict(set)
        for line in open("additional_data/languages/language-spoken-in.csv").readlines()[1:]: # skip header: language,code,country,status
            fields = line.strip().split(",") # only concerned in first two hence ok to read it this way
            country_code = fields[1]
            language_official = fields[0]
            self.lang_spoken_in[language_official].add(country_code)


    def __read_raw_data_dict(self, source_lang, stemmer):
        """
        read in raw UD token information to check for token-presence in source language
        if stemming=False also return char ngram dict
        """
        d_word_freq = defaultdict(int)
        d_stem_freq = defaultdict(int)
        d_char_ngram_freq = defaultdict(int) # only within words
        for line in open("additional_data/ud1.2/{}-ud-all.tok".format(source_lang)):
            word = line.strip() # assume one token per line, separated by newlines
            if word:
                d_word_freq[word] += 1
                stem = stemmer.stem(word)
                d_stem_freq[stem] += 1
                # character 2-3 grams
                for n_gram in get_char_ngrams(word):
                    d_char_ngram_freq[n_gram] += 1
        return d_word_freq, d_stem_freq, d_char_ngram_freq

    def __build_user_vocabs(self):
        with open("additional_data/{}_{}_user_vocabs_incl_dev_test.json".format(self.target_lang, self.source_lang)) as fp:
            indiv_vocab = json.load(fp)
        return indiv_vocab


def load_data(filename, cv=False, train=False,data_source=None):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns a tuple:
        data: a list of InstanceData objects from that data type and track.
        labels: if you specified training data, a dict of instance_id:label pairs, else None
    """

    # 'data' stores a list of 'InstanceData's as values.
    data = []

    # If this is training data, then 'labels' is a dict that contains instance_ids as keys and labels as values.
    training = train
    labels = dict()

    num_exercises = 0
    sys.stderr.write('Loading instances...\n')

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            # If there's nothing in the line, then we're done with the exercise. Print if needed, otherwise continue
            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    sys.stderr.write('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...\n')

            # If the line starts with #, then we're beginning a new exercise
            elif line[0] == '#':
                list_of_exercise_parameters = line[2:].split()
                instance_properties = dict()
                instance_properties["sentence"] = []
                for exercise_parameter in list_of_exercise_parameters:
                    [key, value] = exercise_parameter.split(':')
                    if key == 'countries':
                        value = value.split('|')
                    elif key == 'days':
                        value = float(value)
                    elif key == 'time':
                        if value == 'null':
                            value = None
                        else:
                            assert '.' not in value
                            value = int(value)
                    instance_properties[key] = value

            # Otherwise we're parsing a new Instance for the current exercise
            else:
                line = line.split()
                if training or cv:
                    assert len(line) == 7  #allow for labels in CV mode

                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]

                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training or cv:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                    instance_properties['label'] = label

                """Hector: New additions here"""
                instance_properties["sentence"].append(line[0])
                data.append(InstanceData(instance_properties=instance_properties))
                if data_source:
                    data_source.token_dict[instance_properties['instance_id']]["token"]=line[1]
                    data_source.token_dict[instance_properties['instance_id']]["part_of_speech"]=line[2]


        sys.stderr.write('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    return data, labels


def dictlist_to_file(M,header,filename,sep="\t"):
    fout = open(filename, mode="w")
    for x in M:
        x_out=[str(x.get(h,0)) for h in header]
        fout.write(sep.join(x_out)+"\n")
    fout.close()


def save_features_to_file(args, X_train, Y_train, X_test,sep="\t"):
    train_keys = set()
    for x in X_train:
        train_keys.update(set(x.keys()))
    header = sorted(train_keys)
    dictlist_to_file(X_train,header,args.train+".feats",sep)
    dictlist_to_file(X_test, header, args.test + ".feats",sep)
    fout = open(args.train + ".header", mode="w")
    fout.write(sep.join(header) + "\n")
    fout.close()
    fout = open(args.train + ".labels", mode="w")
    fout.write("\n".join([str(x) for x in Y_train]) + "\n")
    fout.close()
    return


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance. This is meant to act as easy access to the
    data, and provides a launching point for deriving your own features from the data.
    """
    def __init__(self, instance_properties):

        # Parameters specific to this instance
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        # Derived parameters specific to this instance
        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])

        # Derived parameters specific to this exercise
        self.exercise_id = self.instance_id[:10]

        # Parameters shared across the whole session
        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']

        self.sentence = instance_properties['sentence']

        if 'label' in instance_properties:
            self.label = int(instance_properties['label'])
        else:
            self.label = None

        # Derived parameters shared across the whole session
        self.session_id = self.instance_id[:8]

    def to_features(self,data_source, active_feats):
        """
        Prepares those features that we wish to use in the LogisticRegression example in this file. We introduce a bias,
        and take a few included features to use. Note that this dict restructures the corresponding features of the
        input dictionary, 'instance_properties'.

        Returns:
            to_return: a representation of the features we'll use for logistic regression in a dict. A key/feature is a
                key/value pair of the original 'instance_properties' dict, and we encode this feature as 1.0 for 'hot'.
        """
        to_return = dict()

        if "user" in active_feats:
            to_return['user:' + self.user] = 1.0

        if "session" in active_feats:
            to_return['session:' + self.session] = 1.0

        if "client" in active_feats:
            to_return['client:' + self.client] = 1.0

        if "base" in active_feats:
            to_return['format:' + self.format] = 1.0
            to_return['token:' + self.token.lower()] = 1.0
            to_return['token_stem:' + data_source.target_stemmer.stem(self.token.lower())] = 1.0

        if "pos" in active_feats:

            to_return['part_of_speech:' + self.part_of_speech] = 1.0

            """Hector: some new features as per Feb 26"""
            sentence_pos = ["START"] + [data_source.token_dict[t]["part_of_speech"] for t in self.sentence] + ["END"]
            pos_trigram = "+".join([sentence_pos[self.token_index - 1], sentence_pos[self.token_index],
                                    sentence_pos[self.token_index + 1]])
            to_return["pos_trigram:"+pos_trigram] = 1
            # add bigram (word before)
            pos_bigram = "+".join([sentence_pos[self.token_index - 1], sentence_pos[self.token_index]])
            to_return["pos_bigram_-1:"+pos_bigram] = 1
            pos_bigram = "+".join([sentence_pos[self.token_index], sentence_pos[self.token_index + 1]])
            to_return["pos_bigram_+1:"+pos_bigram] = 1

        if "morph" in active_feats:
            for morphological_feature in self.morphological_features:
                if not morphological_feature.startswith("fPOS"):
                    to_return['morphological_feature:' + morphological_feature] = 1.0

        if "dep" in active_feats:
            to_return['dependency_label:' + self.dependency_label] = 1.0
            if "part_of_speech" in data_source.token_dict[self.dependency_edge_head]:
                head_pos = data_source.token_dict[self.dependency_edge_head]["part_of_speech"]
                to_return['dependency_head_pos:' + head_pos] = 1.0
            to_return['dependency_distance:{}'.format(self.token_index-self.dependency_edge_head)] = 1.0

        if "form" in active_feats:
            # check if the token contains symbols that are non ASCII, i.e. most likely diacritics that English-speaking users
            to_return['constains_non_ascii:'] = int(self.token.encode('ascii', 'ignore').decode("utf-8") != self.token)

            """ BP: start adding source language information"""
            trg_token = self.token
            trg_token_stem = data_source.target_stemmer.stem(trg_token)
            if exists_in_dictionary(trg_token, data_source.source_lang_word_freq_dict):
                to_return["token_exists_in_source_lang"] = 1.0  # indicator feature if raw token is present in source language
            if exists_in_dictionary(trg_token_stem, data_source.source_lang_stem_freq_dict):
                to_return["stem_exists_in_source_lang"] = 1.0

            ## proportion of char-n-grams of word which exist in source
            count_ngram_overlap = get_char_ngram_overlap_src_trg(trg_token, data_source)
            if count_ngram_overlap:
                #to_return["char_n_overlap_ratio_binned"] = get_binned_overlap_ratio(count_ngram_overlap)
                to_return["char_n_overlap_ratio_binned={}".format(get_binned_01_range(count_ngram_overlap))] = 1.0

            sentence_tokens = ["START"] + [data_source.token_dict[t]["token"] for t in self.sentence] + ["END"]
            prev_token = sentence_tokens[self.token_index - 1]

            """ BP: new char n-gram divergence feature as per March 3 """
            # add KL divergence of char n-grams src <> trg lang
            kl_div = get_kl_div(trg_token, data_source)
            if kl_div:
                #to_return['kl_binned'] = kl_div
                to_return['kl_binned={}'.format(kl_div)] = 1.0
                
                """cumulative binned KL-divergence"""
                for i in np.arange(0, kl_div + .1, .1):
                    to_return['kl_binned_cumu={}'.format(i)] = 1.0

            """ BP: extend to include prev token KL (ratio less powerful, tested and removed again)"""
            if prev_token != "START":
                kl_div_prev = get_kl_div(prev_token, data_source)
                if kl_div_prev:
                    to_return['kl_binned_prev_token={}'.format(kl_div_prev)] = 1.0
                    
                    """cumulative binned KL-divergence of prev token"""
                    for i in np.arange(0, kl_div_prev + .1, .1):
                        to_return['kl_binned_prev_cumu={}'.format(i)] = 1.0

        if "position" in active_feats:
            to_return['is_last_token'] = int(len(self.sentence) == self.token_index)

        if "country" in active_feats:
            for c in self.countries:
                to_return['country={}'.format(c)] = 1.0
                ## check if src language is spoken in country
                if c in data_source.lang_spoken_in[data_source.target_lang]:
                    to_return['user_in_country_of_target_lang={}'.format(data_source.target_lang)] = 1.0
                    #print(c, data_source.lang_spoken_in[data_source.target_lang])
            if len(self.countries) > 1:
                to_return["present_in_multiple_countries"] = 1.0

        if "len" in active_feats:
            to_return['sentence_length_binned={}'.format(get_binned_sentence_len(len(self.sentence)))] = 1.0

            for i in get_binned_sentence_len(len(self.sentence), cumu=True):
                to_return['sentence_length_binned_cumu={}'.format(i)] = 1.0

        if "time" in active_feats:
            """SK: time features per March 3 and March 10"""
            if type(self.days) == float:
                to_return['days_binned={}'.format((self.days // 4))] = 1.0
                """cumulative bins, time"""
                for i in range(0, (int(self.days) // 4) + 1):
                    to_return['days_binned_cumu={}'.format(i)] = 1.0
            if type(self.time) == int:
                to_return['time_bin_log={}'.format(get_binned_log_time(self.time))] = 1.0
                to_return['tokentime_bin_log={}'.format(get_binned_log_time(self.time / len(self.sentence), per_token=True))] = 1.0

                """cumulative bins, time"""
                for i in get_binned_log_time(self.time, cumu=True):
                    to_return['time_bin_log_cumu={}'.format(i)] = 1.0
                for i in get_binned_log_time((self.time / len(self.sentence)), cumu=True, per_token=True):
                    to_return['tokentime_bin_log_cumu={}'.format(i)] = 1.0


        if "uvocab" in active_feats:
            """SK: features based on the userVocabs available via DataSource"""
            cur_vocab = data_source.user_vocabs[self.user]
            if ((self.token.lower() in cur_vocab.keys()) 
                and (cur_vocab[self.token.lower()]['seen'][0] < self.days)):
                
                errors_prev = [d for d in cur_vocab[self.token.lower()]['errors'] if d < self.days]
                seen_prev = [d for d in cur_vocab[self.token.lower()]['seen'] if d < self.days]
                
                to_return['seen'] = True
                to_return['seen_count={}'.format(get_binned_seen_count(len(seen_prev)))]  = 1.0
                to_return['days_since_seen={}'.format(get_binned_days_since(self.days, seen_prev[-1]))] = 1.0
                to_return['was_error_last={}'.format(seen_prev[-1] in errors_prev)] = 1.0
                token_err_rate_cur = np.round(len(errors_prev) / len(seen_prev), 1)  # ten bins
                to_return['token_error_rate_cur={}'.format(token_err_rate_cur)] = 1.0
                days_since_err = "n/a" if not errors_prev else get_binned_days_since(self.days, errors_prev[-1])
                to_return['days_since_error={}'.format(days_since_err)] = 1.0

                """cumulative bins, uvocab"""
                for i in get_binned_seen_count(len(seen_prev), cumu=True):
                    to_return['seen_count_cumu={}'.format(i)] = 1.0
                for i in get_binned_days_since(self.days, seen_prev[-1], cumu=True):
                    to_return['days_since_seen_cumu={}'.format(i)] = 1.0
                for i in np.arange(0, round(token_err_rate_cur, 1) + .1,.1):
                    to_return['token_error_rate_cur_cumu={}'.format(i)] = 1.0
                if not days_since_err == 'n/a':
                    for i in get_binned_days_since(self.days, errors_prev[-1], cumu=True):
                        to_return['days_since_error_cumu={}'.format(i)] = 1.0

            """Hector: added a simple overall log freq feature"""
            freq = 0
            if self.token.lower() in cur_vocab.keys():
                seen = cur_vocab[self.token.lower()]["seen"]
                freq = math.log(len(seen))
            to_return['user_word_freq_'+self.token.lower()] = freq

        if "comb_format" in active_feats:
            """ combine all features with task type (format) """
            format_str = to_return['format:' + self.format]
            to_return_combination = {}
            for feat in to_return.keys():
                if not feat.startswith("format:"):
                    ### TODO: check if we want to ADD them or to replace them with task-specific submodel
                    if "comb_format_ADD" in active_feats:
                        to_return["{}_{}".format(format_str, feat)] = 1.0
                    else:
                        to_return_combination["{}_{}".format(format_str, feat)] = 1.0
            if not "comb_format_ADD" in active_feats:
                to_return = to_return_combination # override to use only combinations

        if "comb_user" in active_feats:
            """ combine user_ID with task type (format) """
            to_return_combination = {}
            for feat in to_return.keys():
                if not feat.startswith("user:"):
                    to_return_combination["{}_{}".format(self.user, feat)] = 1.0
            to_return = to_return_combination

        return to_return

    def __get_header(self):
        """ helper function for printing """
        ## user:XEinXf5+  countries:CO  days:13.695  client:web  session:lesson  format:reverse_translate  time:8
        country = self.countries[0] if len(self.countries) == 1 else "|".join(self.countries)
        time = self.time if self.time is not None else 'null'
        return "# user:{}\tcountries:{}\tdays:{}\tclient:{}\tsession:{}\tformat:{}\ttime:{}".format(self.user,
                                                                                                   country,
                                                                                                   self.days,
                                                                                                   self.client,
                                                                                                   self.session,
                                                                                                   self.format,
                                                                                                   time)

    def __str__(self):
        """ print out data instance in original format"""
        header = self.__get_header()

        morph_feats = "|".join(["{}={}".format(k,v) for k,v in self.morphological_features.items()])
        # sSLwu6yx0201  I             PRON    Case=Nom|Number=Sing|Person=1|PronType=Prs|fPOS=PRON++PRP       nsubj        3  0
        if self.label is not None:
            instance = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.instance_id, self.token, self.part_of_speech,
                                                      morph_feats, self.dependency_label,
                                                      self.dependency_edge_head, self.label)
        else:
            instance = "{}\t{}\t{}\t{}\t{}\t{}".format(self.instance_id, self.token, self.part_of_speech,
                                                      morph_feats, self.dependency_label,
                                                          self.dependency_edge_head)

        if self.token_index == 1:
            # print header
            return "\n{}\n{}".format(header, instance)
        else:
            return instance
