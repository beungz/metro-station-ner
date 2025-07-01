from scripts.make_dataset import generate_full_sentence_dataset, sample_sentence_dataset, t5_apply_tokenization, crf_preprocess
from scripts.model import t5_train_model, crf_train_model, crf_evaluate_canonical_predictions, naive_evaluate_canonical_predictions


# List of stations (official names), their alternative names, and misspellings
stations = {
    "National Stadium": ["nat stadium", "national sport stadium", "national stadiam", "national stadum", "national studiam", "national statium", "ntl stadium", 
                            "sanam keela", "sports stadium", "stadium bts", "stadium national", "the national stadium"],
    "Siam": ["sayam", "siaam", "siamm", "siyam", "syam"],
    "Ratchadamri": ["rachadamri", "rachadumri", "radchadamri", "rajadamri", "ratchadamli", "ratchadamree", "ratchadumri", "ratchdamri", "ratjadamri"],
    "Sala Daeng": ["sala daeng", "sala dange", "sala deng", "saladaeng", "saladang", "salah daeng"],
    "Chong Nonsi": ["chong non si", "chong nonsie", "chong nonsii", "chong nonsri", "chong nonsy", "chongnonsee", "chongnonsi"],
    "Saint Louis": ["saint luis", "saint louiz", "st louie", "st louis", "st luis", "st. lewis", "st. louis"],
    "Surasak": ["suracak", "surasack", "surasek", "surassak"],
    "Saphan Taksin": ["sapan taksin", "saphan taksen", "saphan takshin", "saphan taksine", "saphan takzin", "saphan taxin", "saphantaksin", "taksin bridge"]
}


# List of templates for model training, with one station placeholder [STATION]
# All words, including "?", must be separated by space, as we use simple white space tokenizer
base_templates = [
    "Can I reach [STATION] from Chulalongkorn University ?",
    "Can I take MRT to get to [STATION] ?",
    "Can I walk from [STATION] to MBK ?",
    "Can I walk to [STATION] ?",
    "How do I get to [STATION] ?",
    "How far is [STATION] from Pratunam ?",
    "How to go from the airport to [STATION] ?",
    "Is [STATION] close to the city center ?",
    "Is [STATION] open at night ?",
    "Is [STATION] open now ?",
    "What BTS line is [STATION] on ?",
    "Where is [STATION] ?",
    "Which exit should I use at [STATION] ?",
    "Which line is [STATION] on ?",
    "How can I get to [STATION] ?",
    "Is it possible to walk to [STATION] from here ?",
    "Is [STATION] served by the BTS ?",
    "Which BTS stop is closest to [STATION] ?",
    "What is the best way to reach [STATION] ?",
    "Do I need to transfer to get to [STATION] ?",
    "Is [STATION] on the Silom or Sukhumvit line ?",
    "Can I travel to [STATION] using the Skytrain ?",
    "How far is [STATION] from central Bangkok ?",
    "What is the location of [STATION] ?",
    "Where can I find [STATION] ?",
    "Is [STATION] near popular attractions ?",
    "Does BTS stop at [STATION] ?",
    "What is the fastest way to reach [STATION] ?",
    "Can I access [STATION] via BTS ?"
]


# List of templates for model training, with two station placeholder, [STATION_1] and [STATION_2]
# All words, including "?", must be separated by space, as we use simple white space tokenizer
# [STATION_1] must come first, followed by [STATION_2]
two_station_templates = [
    "How to go from [STATION_1] to [STATION_2] ?",
    "Can I ride BTS from [STATION_1] to [STATION_2] directly ?",
    "Is there a transfer between [STATION_1] and [STATION_2] ?",
    "Which station do I change at when going from [STATION_1] to [STATION_2] ?",
    "How many stops between [STATION_1] and [STATION_2] ?",
    "How long does it take to go from [STATION_1] to [STATION_2] ?",
    "Does [STATION_1] connect to [STATION_2] ?",
    "What is the BTS route from [STATION_1] to [STATION_2] ?",
    "Is [STATION_1] on the same line as [STATION_2] ?",
    "How many minutes between [STATION_1] and [STATION_2] by BTS ?",
    "Which line should I take from [STATION_1] to reach [STATION_2] ?",
    "Do I need to change lines from [STATION_1] to [STATION_2] ?",
    "Is there a direct BTS line connecting [STATION_1] and [STATION_2] ?",
    "Can I go from [STATION_1] to [STATION_2] without switching ?",
    "What is the fastest way to get from [STATION_1] to [STATION_2] by Skytrain ?",
    "Is it possible to travel from [STATION_1] to [STATION_2] without transfer ?",
    "Which interchange connects [STATION_1] to [STATION_2] ?",
    "How many stations are between [STATION_1] and [STATION_2] ?",
    "Where do I need to transfer to go from [STATION_1] to [STATION_2] ?",
    "How do I commute from [STATION_1] to [STATION_2] using BTS ?"
]


# Prefixes and suffixes to add before/after station name
prefixes = ["", "BTS ", "Skytrain "]
suffixes = ["", " station", " BTS", " BTS stop"]


# List of keys, with their neighbors, to be used in generation of corrupt station names
keyboard_neighbors = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx'
}


# List of characters, with their alternative phonetics, to be used in generation of corrupt station names
phonetic_rules = {
    "ph": "f", "ch": "sh", "sh": "ch", "r": "l", "l": "r",
    "d": "t", "t": "d", "k": "g", "g": "k"
}


def main():
    '''Main function to run the entire pipeline for detecting Bangkok Metro Station Name from text (Named Entity Recognition).'''
    # A. Get dataset
    print("\nA. Get dataset")

    # Step A1. Generate full sentence dataset
    print("\nStep A1. Generate full sentence dataset")
    df_train, df_test, df_train_bio, df_test_bio = generate_full_sentence_dataset(stations, base_templates, two_station_templates, prefixes, suffixes, keyboard_neighbors, phonetic_rules)

    # Step A2. Shuffle and randomly sample the full sentence dataset to get train, validation, and test set
    print("\nStep A2. Shuffle and randomly sample the full sentence dataset to get train, validation, and test set")
    train_sample_size = 8000
    val_sample_size = 1000
    test_sample_size =1000

    df_train_small, df_val_small, df_test_small, df_train_bio_small, df_test_bio_small = sample_sentence_dataset(df_train, df_test, df_train_bio, df_test_bio, train_sample_size, val_sample_size, test_sample_size)

    # Step A3. T5: preprocess train data
    print("\nStep A3. T5: preprocess train data")
    train_ds, eval_ds, test_ds = t5_apply_tokenization(df_train_small, df_val_small, df_test_small)

    # Step A4. CRF: preprocess train data and build features
    print("\nStep A4. CRF: preprocess train data and build features")
    X_train, X_test, X_test_sentence, y_train, y_test = crf_preprocess(df_train_bio_small, df_test_bio_small)

    # B. Train T5 and CRF models
    print("\nB. Train T5 and CRF models")

    # Step B1: Train T5 model
    print("\nStep B1: Train T5 model")
    t5_model, t5_tokenizer, t5_test_precision, t5_test_recall, t5_test_f1, t5_test_accuracy = t5_train_model(train_ds, eval_ds, test_ds)

    # Step B2: Train CRF model
    print("\nStep B2: Train CRF model")

    # Define the list of hyperparameter for cross validation
    param_grid = {
        'c1': [0.001, 0.01, 0.1, 1, 10, 100],
        'c2': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    crf_model, crf_test_precision, crf_test_recall, crf_test_f1, crf_test_accuracy = crf_train_model(X_train, X_test, y_train, y_test, param_grid)

    crf_canon_test_precision, crf_canon_test_recall, crf_canon_test_f1, crf_canon_test_accuracy = crf_evaluate_canonical_predictions(crf_model, X_test, X_test_sentence, y_test)

    # Step B3: Evaluate naive model
    print("\nStep B3: Evaluate naive model")
    naive_test_precision, naive_test_recall, naive_test_f1, naive_test_accuracy = naive_evaluate_canonical_predictions(stations, df_test_small)


if __name__ == "__main__":
    main()