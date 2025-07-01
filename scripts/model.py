import pandas as pd
import numpy as np

from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV

from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, EvalPrediction, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

from scripts.build_features import sent2features
from scripts.make_dataset import simple_tokenize

from fuzzywuzzy import fuzz

import re
import joblib
import os

# Functions used for T5/CRF model training, prediction, and metric evaluation



# Label list for crf_f1_score, which is used as evaluation metrics function for GridSearchCV of CRF model
labels = [
    'O',
    'B-National_Stadium', 'I-National_Stadium',
    'B-Siam', 'I-Siam',
    'B-Ratchadamri', 'I-Ratchadamri',
    'B-Sala_Daeng', 'I-Sala_Daeng',
    'B-Chong_Nonsi', 'I-Chong_Nonsi',
    'B-Saint_Louis', 'I-Saint_Louis',
    'B-Surasak', 'I-Surasak',
    'B-Saphan_Taksin', 'I-Saphan_Taksin'
]



def t5_train_model(train_ds, eval_ds, test_ds):
    '''Train T5 model'''

    # Load tokenizer and model
    model_name = "t5-base"    # t5-base or t5-small
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Test run on CPU
    # device = torch.device("cpu")
    # model = model.to(device)

    # Move model to GPU
    model = model.to("cuda")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/deep_learning",
        eval_strategy="epoch",              #no, epoch
        save_strategy="epoch",              #no, epoch
        save_total_limit=2,
        logging_dir="./models/deep_learning/logs",
        logging_steps=20,
        per_device_train_batch_size=24,     # 4, 8, 16, 24
        per_device_eval_batch_size=24,      # 4, 8, 16, 24
        gradient_accumulation_steps=1,      # 4, 1
        learning_rate=2e-4,                 # 2e-5
        num_train_epochs=3,                 # 3, 5, 10
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        torch_compile = False,
        report_to=[],                       # [], "none"
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=1
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=t5_compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model and its tokenizer
    t5_final_model_path = os.path.join("models", "deep_learning")
    trainer.save_model(t5_final_model_path)
    tokenizer.save_pretrained(t5_final_model_path)

    # Evaluation on Validation Set
    val_results = trainer.evaluate(eval_dataset=eval_ds)
    val_precision = val_results["eval_precision"]
    val_recall = val_results["eval_recall"]
    val_f1 = val_results["eval_f1"]
    val_accuracy = val_results["eval_accuracy"]

    print("\nEvaluation on Validation Set: ")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1 Score:  {val_f1:.4f}")
    print(f"Accuracy:  {val_accuracy:.4f}")

    # Evaluation on Test Set
    test_results = trainer.evaluate(eval_dataset=test_ds)
    test_precision = test_results["eval_precision"]
    test_recall = test_results["eval_recall"]
    test_f1 = test_results["eval_f1"]
    test_accuracy = test_results["eval_accuracy"]

    print("\nEvaluation on Test Set: ")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")

    return trainer.model, tokenizer, test_precision, test_recall, test_f1, test_accuracy



def t5_compute_metrics(p: EvalPrediction):
    '''Compute evaluation metrics for T5 model training'''

    # Tokenizer for T5
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    predictions = p.predictions
    labels = p.label_ids

    # Decode to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Split by ";" and convert to sets
    preds_sets = [set(pred.split(" ; ")) for pred in decoded_preds]
    labels_sets = [set(label.split(" ; ")) for label in decoded_labels]

    # Calculate precision, recall, f1, accuracy
    precisions = []
    recalls = []
    f1s = []
    accuracies = []

    for pred_set, label_set in zip(preds_sets, labels_sets):
        true_positives = len(pred_set.intersection(label_set))
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(label_set) if label_set else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        accuracy = 1.0 if pred_set == label_set else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    avg_accuracy = sum(accuracies) / len(accuracies)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "accuracy": avg_accuracy,
    }



def t5_predict_station(model, tokenizer, text):
    '''Use T5 model to predict station name(s)'''
    input_text = f"find station: {text}"
    # Tokenize text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Get prediction from the model
    output = model.generate(input_ids, max_length=32)
    return tokenizer.decode(output[0], skip_special_tokens=True)



def crf_train_model(X_train, X_test, y_train, y_test, param_grid):
    '''Train CRF model'''

    # Initiate CRF model
    crf_model = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=crf_model, param_grid=param_grid, cv=3, scoring=crf_f1_score, verbose=2, n_jobs=1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_train, y_train)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    val_f1 = grid_search.best_score_
    print("Best Cross-validation F1 score:: {:.6f}".format(val_f1))

    # Save the best model
    best_crf_model = grid_search.best_estimator_
    crf_final_model_path = os.path.join("models", "classical_machine_learning", "best_crf_model.joblib")
    joblib.dump(best_crf_model, crf_final_model_path)

    # Evaluate the best model
    # Predict on the test set
    y_pred = best_crf_model.predict(X_test)

    # Flatten lists for evaluation
    y_true_flat = [tag for sent in y_test for tag in sent]
    y_pred_flat = [tag for sent in y_pred for tag in sent]

    # Evaluation on Test Set at Token-level: Treat B-station and I-station separately
    print("\nEvaluation on Test Set at Token-level: Treat B-station and I-station separately")

    print("\nClassification report:")
    print(classification_report(y_true_flat, y_pred_flat, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true_flat, y_pred_flat))

    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    test_accuracy = accuracy_score(y_true_flat, y_pred_flat)

    print(f"\nPrecision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 score:  {test_f1:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")

    return best_crf_model, test_precision, test_recall, test_f1, test_accuracy



def crf_f1_score(estimator, X, y):
    '''Calculate F1 score to be used as evaluation metrics for CRF model training'''
    try:
        y_pred = estimator.predict(X)

        # Compute F1 score
        score = metrics.flat_f1_score(y, y_pred, average='weighted', labels=labels, zero_division=0)
        if np.isnan(score):
            return 0.0
        return score
    except Exception as e:
        print(f"Warning: scoring exception: {e}", flush=True)
        return 0.0



def bio_to_canonical_stations(tokens, bio_tags):
    '''Convert bio-tags of station into canonical station name'''

    # List to collect station names
    stations = []
    current_station_tokens = []
    current_station_label = None

    # Loop through each token and its corresponding BIO tag
    for token, tag in zip(tokens, bio_tags):
        if tag == 'O':
            if current_station_tokens:
                # If there is previous labels in progress, then append labels to stations list
                station_name = " ".join(current_station_tokens)
                stations.append(current_station_label)
                current_station_tokens = []
                current_station_label = None
            continue
        
        # Split BIO tag by '-' into type (B/I) and station name
        tag_type = tag.split('-')[0]
        tag_label = tag.split('-')[1]

        if tag_type == 'B':
            # Tag B indicates the start of station name
            if current_station_tokens:
                # If there is previous labels in progress, then append labels to stations list
                stations.append(current_station_label)
            # Start collecting token of new station
            current_station_tokens = [token]
            current_station_label = tag_label.replace('_', ' ')
        elif tag_type == 'I' and current_station_label == tag_label.replace('_', ' '):
            # Tag I indicates parts of station name following the B-tag
            # Continue collecting token of current station
            current_station_tokens.append(token)
        else:
            # Tag sequence error or new start (treat as B)
            if current_station_tokens:
                stations.append(current_station_label)
            current_station_tokens = [token]
            current_station_label = tag_label.replace('_', ' ')

    if current_station_tokens:
        # If there is previous labels in progress, then append labels to stations list
        stations.append(current_station_label)

    # Remove duplicates
    seen = set()
    canonical_stations = []
    for s in stations:
        if s not in seen:
            canonical_stations.append(s)
            seen.add(s)

    # Join with ' ; ' if multiple stations found
    return " ; ".join(canonical_stations) if canonical_stations else ""



def crf_predict_station(model, text):
    '''Use CRF model to predict station name(s)'''
    # Tokenize text
    tokens = simple_tokenize(text)
    # Build features based on text
    features = sent2features(tokens)
    # Get prediction in forms of bio-tags
    predicted_bio = model.predict([features])[0]
    # Convert from bio-tags into canonical output (simple station name(s))
    canonical_output = bio_to_canonical_stations(tokens, predicted_bio)
    return canonical_output



def convert_bio_to_canonicals(y_bio, sentences):
    '''Convert bio-tags of station into canonical station name'''
    canonical_labels = []
    for bio_tags, tokens in zip(y_bio, sentences):
        label = bio_to_canonical_stations(tokens, bio_tags)
        canonical_labels.append(label)
    return canonical_labels


def canonicalize_station_set(canon_str):
    '''Normalize canonical string by removing whitespace, lowercase, sorting'''
    return " ; ".join(sorted(set(
        s.strip().lower() for s in canon_str.split(";") if s.strip()
    )))



def crf_evaluate_canonical_predictions(model, X_test, X_test_sentence, y_test):
    '''Evaluate CRF model on its prediction in forms of canonical station name(s)'''

    # Get prediction in forms of bio-tags
    y_pred = model.predict(X_test)

    # Convert true and predicted BIO tags to canonical station strings
    y_true_canonicals = convert_bio_to_canonicals(y_bio=y_test, sentences=X_test_sentence)
    y_pred_canonicals = convert_bio_to_canonicals(y_bio=y_pred, sentences=X_test_sentence)

    # Normalize both true and predicted canonical station strings
    y_true_norm = [canonicalize_station_set(y) for y in y_true_canonicals]
    y_pred_norm = [canonicalize_station_set(y) for y in y_pred_canonicals]

    # Evaluation on Test Set: Exact Canonical Match
    print("\nEvaluation on Test Set: Exact Canonical Match")

    # Classification report
    print("\nClassification Report")
    print(classification_report(y_true_norm, y_pred_norm, zero_division=0))

    # Confusion matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_true_norm, y_pred_norm))

    # Standard metrics
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_true_norm, y_pred_norm, average='weighted', zero_division=0)
    test_accuracy = accuracy_score(y_true_norm, y_pred_norm)

    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")

    return test_precision, test_recall, test_f1, test_accuracy



def naive_predict_station(stations, text, threshold=90, return_bio=False):
    '''Use naive string matching and fuzzy matching to detect station mentions in text.'''

    # Add canonical names as variants in lowercase
    for canonical in list(stations.keys()):
        stations[canonical].append(canonical.lower())

    # Lowercase and tokenize text
    text_lower = text.lower()
    tokens = simple_tokenize(text)

    # Track start/end spans for each token
    token_spans = []
    pos = 0
    for tok in tokens:
        start = text_lower.find(tok, pos)
        token_spans.append((start, start + len(tok)))
        pos = start + len(tok)

    # List of matched station
    found_mentions = []

    # Search for each station variant in the text
    for canonical, variants in stations.items():
        for variant in variants:
            variant_lower = variant.lower()
            if variant_lower in text_lower:
                # Direct match
                start_idx = text_lower.find(variant_lower)
                end_idx = start_idx + len(variant_lower)
                found_mentions.append((canonical, start_idx, end_idx))
            else:
                # Fuzzy match
                score = fuzz.partial_ratio(variant_lower, text_lower)
                if score >= threshold:
                    match = re.search(re.escape(variant_lower), text_lower)
                    if match:
                        found_mentions.append((canonical, match.start(), match.end()))

    # Remove overlapping spans, keep max length
    found_mentions.sort(key=lambda x: (x[1], -x[2]))
    non_overlap = []
    used = [False] * len(text)
    for canon, s, e in found_mentions:
        if not any(used[s:e]):
            non_overlap.append((canon, s, e))
            for i in range(s, e):
                used[i] = True
        if len(non_overlap) == 2:
            break
    
    # Generate BIO-tag
    if return_bio:
        # Initialize all tags with "O"
        tags = ['O'] * len(tokens)

        # Assign bio-tags to tokens
        for canon, span_start, span_end in non_overlap:
            for i, (tok_start, tok_end) in enumerate(token_spans):
                if tok_start >= span_start and tok_end <= span_end:
                    tags[i] = f'I-{canon.replace(" ", "_")}'
            for i, tag in enumerate(tags):
                if tag.startswith('I-'):
                    tags[i] = tag.replace('I-', 'B-', 1)
                    break
        return tags
    else:
        return ' ; '.join([canon for canon, _, _ in non_overlap])



def normalize(text):
    # Normalize strings for comparison
    return ";".join(sorted([s.strip() for s in text.lower().split(";")])) if isinstance(text, str) else ""



def naive_evaluate_canonical_predictions(stations, df_test_small):
    '''Evaluate naive model on its prediction in forms of canonical station name(s)'''

    # Get ground truth station names
    y_true_raw = df_test_small['target_text'].tolist()

    # Predict canonical names using naive_predict_station (return_bio=False)
    y_pred_raw = [naive_predict_station(stations, text) for text in df_test_small['input_text']]

    y_true = [normalize(s) for s in y_true_raw]
    y_pred = [normalize(s) for s in y_pred_raw]

    # Evaluation on Test Set: Exact Canonical Match
    print("\nEvaluation on Test Set: Exact Canonical Match")

    # Classification report
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    # Standard metrics
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")

    return test_precision, test_recall, test_f1, test_accuracy

