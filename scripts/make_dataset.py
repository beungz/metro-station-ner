import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer
import torch
import random
from itertools import permutations
from sklearn.utils import shuffle
from scripts.build_features import sent2features
import os

# Functions used to create dataset and preprocess before model training



def corrupt_word(original, keyboard_neighbors, phonetic_rules, n=5):
    """Generate up to n misspellings of a station name using keyboard, phonetic, deletion, and insertion errors."""

    keep_space = " " in original
    spaced_form = original
    compact_form = original.replace(" ", "").lower()

    variants = set()
    attempts = 0

    # Misspelling Generator:Loop to generate variants up to n variants
    while len(variants) < n and attempts < 50:
        attempts += 1
        corrupted = list(compact_form)

        # Randomly choose a method to create misspellings
        method = random.choice(["keyboard", "phonetic", "delete", "insert", "none"])

        # Keyboard typo (pressing wrong nearby key)
        if method == "keyboard":
            i = random.randint(0, len(corrupted) - 1)
            ch = corrupted[i]
            if ch in keyboard_neighbors:
                corrupted[i] = random.choice(keyboard_neighbors[ch])

        # Substitute with similar phonetics
        elif method == "phonetic":
            corrupted_str = ''.join(corrupted)
            for orig, repl in phonetic_rules.items():
                if orig in corrupted_str:
                    corrupted_str = corrupted_str.replace(orig, repl, 1)
                    corrupted = list(corrupted_str)
                    break
        
        # Randomly delete one character
        elif method == "delete" and len(corrupted) > 4:
            i = random.randint(1, len(corrupted) - 2)
            del corrupted[i]

        # Randomly insert one vowel character
        elif method == "insert" and len(corrupted) < 20:
            i = random.randint(1, len(corrupted) - 1)
            corrupted.insert(i, random.choice("aeiou"))

        # Recombine and add space (50% probability)
        corrupted_str = ''.join(corrupted)
        if keep_space and random.random() < 0.5:
            space_pos = spaced_form.find(" ")
            corrupted_str = corrupted_str[:space_pos] + " " + corrupted_str[space_pos:]

        # Add to set if it's different from the original, and have proper length
        if corrupted_str != compact_form and 3 < len(corrupted_str) < len(compact_form) + 3:
            variants.add(corrupted_str)

    return list(variants)



def simple_tokenize(text):
    '''Simple whitespace tokenizer'''
    return text.strip().split()



def bio_tag_station(variant, canonical):
    '''Generate BIO tags for a station variant: B-stationname for the first tag, and I-stationname for the following tags'''
    
    # Tokenize variant of station
    variant_tokens = simple_tokenize(variant)
    # Initialize all tags with "O"
    bio_tags = ["O"] * len(variant_tokens)

    # Replace space with underscore for station name with space(s)
    canonical_no_space = canonical.replace(" ","_")

    # Tag the variant tokens with B- and I-
    # First tag is B-stationname
    bio_tags[0] = f"B-{canonical_no_space}"

    for j in range(1, len(variant_tokens)):
        # Remaining tags are I-stationname
        bio_tags[j] = f"I-{canonical_no_space}"

    return variant_tokens, bio_tags



def generate_station_variants(canonical_name, prefixes, suffixes):
    '''Generate station name variants by adding prefixes and/or suffixes'''
    variants = set()

    # Add prefixes
    for pre in prefixes:
        variant = f"{pre}{canonical_name}".strip()
        variants.add(variant)

    # Add suffixes
    for suf in suffixes:
        variant = f"{canonical_name}{suf}".strip()
        variants.add(variant)

    return list(variants)



def apply_all_variants(canonical, original_variants, prefixes, suffixes, keyboard_neighbors, phonetic_rules):
    '''Generate all possible variants for a station name, including corruptions, lower/upper case, and adding prefix/suffix'''
    
    # Generate variants with corruption and lower/upper
    all_vars = set(original_variants + 
                   corrupt_word(canonical, keyboard_neighbors, phonetic_rules) + 
                   [canonical.lower().replace(" ", ""), canonical.upper()])
    
    # Generate variants with prefix/suffix
    for var in list(all_vars):
        all_vars.update(generate_station_variants(var, prefixes, suffixes))

    return list(all_vars)



def generate_full_sentence_dataset(stations, base_templates, two_station_templates, prefixes, suffixes, keyboard_neighbors, phonetic_rules):
    '''Generate dataset for training/test. This create large dataset, and the sampling/shuffling will be done in sample_sentence_dataset()'''

    # Shuffle templates, both one and two stations
    random.seed(42)
    random.shuffle(base_templates)
    random.shuffle(two_station_templates)

    # Split 80% for train, 20% for test, by splitting templates used to generate dataset into 2 groups. This is to make sure that templates used in test set will be unseen by the model.
    split_ratio = 0.8
    train_base_templates = base_templates[:int(len(base_templates)*split_ratio)]
    test_base_templates = base_templates[int(len(base_templates)*split_ratio):]

    train_two_station_templates = two_station_templates[:int(len(two_station_templates)*split_ratio)]
    test_two_station_templates = two_station_templates[int(len(two_station_templates)*split_ratio):]

    # rows for train/test set, to be converted into dataframes later
    rows_train = []
    rows_test = []
    bio_rows_train = []
    bio_rows_test = []

    # One-station training/test data
    for canonical, variants in stations.items():
        # Create variants of station name by applying corruption, lower/upper, prefix/suffix
        all_variants = apply_all_variants(canonical, variants, prefixes, suffixes, keyboard_neighbors, phonetic_rules)
        
        # Generate variants of station name based on templates in train_base_template
        for template in train_base_templates:
            # Tokenize sentence template
            tokens = simple_tokenize(template)
            # Initialize all tags for sentence template, with "O"
            bio_tags = ["O"] * len(tokens)

            # Find location of the station placeholder in tokens
            station_loc = 0
            for i in range(len(tokens)):
                if tokens[i] == '[STATION]':
                    station_loc = i
                    break

            for v in all_variants:
                # Replace station placeholder in sentence template with station variant, to create new sentence
                sentence_without_findstation = f"{template.replace('[STATION]', v)}"
                sentence = f"find station: {sentence_without_findstation}"

                # Generate BIO tags for station variant
                variant_tokens, variant_bio_tags = bio_tag_station(v, canonical)

                # Add "O" to the initialized bio tags (of the sentence template), to accomodate station name with more than one tag
                bio_tags_new = bio_tags + ["O"] * (len(variant_tokens) - 1)

                # Replace "O" tags in sentence with correct station bio tags
                for j in range(len(variant_tokens)):
                    bio_tags_new[station_loc + j] = variant_bio_tags[j]

                # Append sentence + canonical station name to dataset for T5, and sentence without "find station" + bio tag to dataset for CRF
                rows_train.append({
                    "input_text": sentence,
                    "target_text": canonical
                })
                bio_rows_train.append({
                    "input_text_bio": sentence_without_findstation,
                    "bio_tags": " ".join(bio_tags_new)
                })

        # Generate variants of station name based on templates in test_base_template
        for template in test_base_templates:
            # Tokenize sentence template
            tokens = simple_tokenize(template)
            # Initialize all tags for sentence template, with "O"
            bio_tags = ["O"] * len(tokens)

            # Find location of the station placeholder in tokens
            station_loc = 0
            for i in range(len(tokens)):
                if tokens[i] == '[STATION]':
                    station_loc = i
                    break
            
            for v in all_variants:
                # Replace station placeholder in sentence template with station variant, to create new sentence
                sentence_without_findstation = f"{template.replace('[STATION]', v)}"
                sentence = f"find station: {sentence_without_findstation}"

                # Generate BIO tags for station variant
                variant_tokens, variant_bio_tags = bio_tag_station(v, canonical)

                # Add "O" to the initialized bio tags (of the sentence template), to accomodate station name with more than one tag
                bio_tags_new = bio_tags + ["O"] * (len(variant_tokens) - 1)

                # Replace "O" tags in sentence with correct station bio tags
                for j in range(len(variant_tokens)):
                    bio_tags_new[station_loc + j] = variant_bio_tags[j]

                # Append sentence + canonical station name to dataset for T5, and sentence without "find station" + bio tag to dataset for CRF
                rows_test.append({
                    "input_text": sentence,
                    "target_text": canonical
                })
                bio_rows_test.append({
                    "input_text_bio": sentence_without_findstation,
                    "bio_tags": " ".join(bio_tags_new)
                })

    # Two-station training/test data
    station_names = list(stations.keys())
    # Create pairs of station names
    canonical_pairs = list(permutations(station_names, 2))

    for canon1, canon2 in canonical_pairs:
        # Create variants of station name by applying corruption, lower/upper, prefix/suffix
        variants1 = apply_all_variants(canon1, stations[canon1], prefixes, suffixes, keyboard_neighbors, phonetic_rules)
        variants2 = apply_all_variants(canon2, stations[canon2], prefixes, suffixes, keyboard_neighbors, phonetic_rules)

        for v1 in variants1:
            for v2 in variants2:
                # Generate variants of station name based on templates in train_two_station_templates
                for template in train_two_station_templates:
                    # Tokenize sentence template
                    tokens = simple_tokenize(template)
                    # Initialize all tags for sentence template, with "O"
                    bio_tags = ["O"] * len(tokens)

                    # Find location of the station placeholder in tokens
                    station_loc_1 = 0
                    station_loc_2 = 0
                    for i1 in range(len(tokens)):
                        if tokens[i1] == '[STATION_1]':
                            station_loc_1 = i1
                            break
                    for i2 in range(len(tokens)):
                        if tokens[i2] == '[STATION_2]':
                            station_loc_2 = i2
                            break

                    # Generate BIO tags for station variant
                    variant_tokens_1, variant_bio_tags_1 = bio_tag_station(v1, canon1)
                    variant_tokens_2, variant_bio_tags_2 = bio_tag_station(v2, canon2)

                    # Add "O" to the initialized bio tags (of the sentence template), to accomodate station name with more than one tag
                    bio_tags_new = bio_tags + ["O"] * (len(variant_tokens_1) + len(variant_tokens_2) - 2)

                    # Replace "O" tags in sentence with correct station bio tags
                    for j1 in range(len(variant_tokens_1)):
                        bio_tags_new[station_loc_1 + j1] = variant_bio_tags_1[j1]
                    station_loc_2 = station_loc_2 + len(variant_tokens_1) - 1
                    for j2 in range(len(variant_tokens_2)):
                        bio_tags_new[station_loc_2 + j2] = variant_bio_tags_2[j2]
                    
                    # Replace station placeholder in sentence template with station variant, to create new sentence
                    sentence_without_findstation = f"{template.replace('[STATION_1]', v1).replace('[STATION_2]', v2)}"
                    sentence = f"find station: {sentence_without_findstation}"

                    # Append sentence + canonical station name to dataset for T5, and sentence without "find station" + bio tag to dataset for CRF
                    rows_train.append({
                        "input_text": sentence,
                        "target_text": f"{canon1} ; {canon2}"
                    })
                    bio_rows_train.append({
                        "input_text_bio": sentence_without_findstation,
                        "bio_tags": " ".join(bio_tags_new)
                    })
                
                # Generate variants of station name based on templates in test_two_station_templates
                for template in test_two_station_templates:
                    # Tokenize sentence template
                    tokens = simple_tokenize(template)
                    # Initialize all tags for sentence template, with "O"
                    bio_tags = ["O"] * len(tokens)

                    # Find location of the station placeholder in tokens
                    station_loc_1 = 0
                    station_loc_2 = 0
                    for i1 in range(len(tokens)):
                        if tokens[i1] == '[STATION_1]':
                            station_loc_1 = i1
                            break
                    for i2 in range(len(tokens)):
                        if tokens[i2] == '[STATION_2]':
                            station_loc_2 = i2
                            break
                    
                    # Generate BIO tags for station variant
                    variant_tokens_1, variant_bio_tags_1 = bio_tag_station(v1, canon1)
                    variant_tokens_2, variant_bio_tags_2 = bio_tag_station(v2, canon2)

                    # Add "O" to the initialized bio tags (of the sentence template), to accomodate station name with more than one tag
                    bio_tags_new = bio_tags + ["O"] * (len(variant_tokens_1) + len(variant_tokens_2) - 2)

                    # Replace "O" tags in sentence with correct station bio tags
                    for j1 in range(len(variant_tokens_1)):
                        bio_tags_new[station_loc_1 + j1] = variant_bio_tags_1[j1]
                    station_loc_2 = station_loc_2 + len(variant_tokens_1) - 1
                    for j2 in range(len(variant_tokens_2)):
                        bio_tags_new[station_loc_2 + j2] = variant_bio_tags_2[j2]

                    # Replace station placeholder in sentence template with station variant, to create new sentence
                    sentence_without_findstation = f"{template.replace('[STATION_1]', v1).replace('[STATION_2]', v2)}"
                    sentence = f"find station: {sentence_without_findstation}"
                    
                    # Append sentence + canonical station name to dataset for T5, and sentence without "find station" + bio tag to dataset for CRF
                    rows_test.append({
                        "input_text": sentence,
                        "target_text": f"{canon1} ; {canon2}"
                    })
                    bio_rows_test.append({
                        "input_text_bio": sentence_without_findstation,
                        "bio_tags": " ".join(bio_tags_new)
                    })

    # Drop duplicates from dataset
    df_train = pd.DataFrame(rows_train).drop_duplicates()
    df_test = pd.DataFrame(rows_test).drop_duplicates()
    df_train_bio = pd.DataFrame(bio_rows_train).drop_duplicates()
    df_test_bio = pd.DataFrame(bio_rows_test).drop_duplicates()

    # Save full dataset to CSV
    full_sentence_dataset_path = os.path.join("data", "processed")
    df_train.to_csv(os.path.join(full_sentence_dataset_path, "bts_train_data.csv"), index=False)
    df_test.to_csv(os.path.join(full_sentence_dataset_path, "bts_test_data.csv"), index=False)
    df_train_bio.to_csv(os.path.join(full_sentence_dataset_path, "bts_train_data_bio.csv"), index=False)
    df_test_bio.to_csv(os.path.join(full_sentence_dataset_path, "bts_test_data_bio.csv"), index=False)

    print(f"Saved {len(df_train)} training and {len(df_test)} test set of full sentence dataset to data\processed folder.")

    return df_train, df_test, df_train_bio, df_test_bio



def sample_sentence_dataset(df_train, df_test, df_train_bio, df_test_bio, train_sample_size, val_sample_size, test_sample_size, random_state=42):
    '''Given full dataset, shuffling and sampling to get smaller version to be used for train/validation/test'''

    # Shuffle dataset
    df_train = shuffle(df_train, random_state=random_state).reset_index(drop=True)
    df_test = shuffle(df_test, random_state=random_state).reset_index(drop=True)

    df_train_bio = shuffle(df_train_bio, random_state=random_state).reset_index(drop=True)
    df_test_bio = shuffle(df_test_bio, random_state=random_state).reset_index(drop=True)

    # Reduce size of the dataset to the sample_size
    df_train_small = df_train[:train_sample_size]
    df_val_small = df_train[-val_sample_size:]
    df_test_small = df_test[:test_sample_size]

    df_train_bio_small = df_train_bio[:train_sample_size + val_sample_size]
    df_test_bio_small = df_test_bio[:test_sample_size]

    # Save sampled dataset to CSV
    sampled_sentence_dataset_path = os.path.join("data", "outputs")
    df_train_small.to_csv(os.path.join(sampled_sentence_dataset_path, "bts_train_data_small.csv"), index=False)
    df_val_small.to_csv(os.path.join(sampled_sentence_dataset_path, "bts_val_data_small.csv"), index=False)
    df_test_small.to_csv(os.path.join(sampled_sentence_dataset_path, "bts_test_data_small.csv"), index=False)
    df_train_bio_small.to_csv(os.path.join(sampled_sentence_dataset_path, "bts_train_data_bio_small.csv"), index=False)
    df_test_bio_small.to_csv(os.path.join(sampled_sentence_dataset_path, "bts_test_data_bio_small.csv"), index=False)

    print("Saved smaller version of train, validation, and test set to data\outputs folder.")

    return df_train_small, df_val_small, df_test_small, df_train_bio_small, df_test_bio_small



def t5_apply_tokenization(df_train_small, df_val_small, df_test_small):
    '''Preprocess data for T5 model'''

    # T5 Tokenizer (t5-base)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def t5_preprocess(text_sample):
        '''Tokenize sentence and target (station name) with T5 tokenizer'''
        input_enc = tokenizer(text_sample["input_text"], padding="max_length", truncation=True, max_length=64)
        target_enc = tokenizer(text_sample["target_text"], padding="max_length", truncation=True, max_length=64)
        input_enc["labels"] = target_enc["input_ids"]
        return input_enc

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df_train_small)
    val_dataset = Dataset.from_pandas(df_val_small)
    test_dataset = Dataset.from_pandas(df_test_small)

    # Apply tokenization, and get the final dataset ready for training
    train_ds = train_dataset.map(t5_preprocess, remove_columns=train_dataset.column_names)
    eval_ds = val_dataset.map(t5_preprocess, remove_columns=val_dataset.column_names)
    test_ds = test_dataset.map(t5_preprocess, remove_columns=val_dataset.column_names)

    print("Data is preprocessed and ready for T5 model training.")

    return train_ds, eval_ds, test_ds



def crf_preprocess(df_train_bio_small, df_test_bio_small):
    '''Preprocess data for CRF model'''
    
    # Prepare train data to be used in GridSearchCV
    train_sents = []
    train_tags = []
    # Tokenize sentence and bio tags
    for _, row in df_train_bio_small.iterrows():
        input_text = row["input_text_bio"]
        bio_tags_text = row["bio_tags"]
        tokens = simple_tokenize(input_text)
        tags = simple_tokenize(bio_tags_text)
        train_sents.append(tokens)
        train_tags.append(tags)

    # Prepare test data
    test_sents = []
    test_tags = []
    # Tokenize sentence and bio tags
    for _, row in df_test_bio_small.iterrows():
        input_text = row["input_text_bio"]
        bio_tags_text = row["bio_tags"]
        tokens = simple_tokenize(input_text)
        tags = simple_tokenize(bio_tags_text)
        test_sents.append(tokens)
        test_tags.append(tags)

    # Extract features from sentence
    X_train = [sent2features(s) for s in train_sents]
    X_test = [sent2features(s) for s in test_sents]
    X_test_sentence = test_sents
    y_train = train_tags
    y_test = test_tags

    print("Data is preprocessed and ready for CRF model training.")

    return X_train, X_test, X_test_sentence, y_train, y_test