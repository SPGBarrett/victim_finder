#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  model_eval.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2021/10/6 8:21
@Version    :  1.0
@Description:  None.

Example:
    Some examples of usage here.
Attributes:
    Attribute description here.
Todo:
    * For module TODOs

'''

# import lib
from bzrs_main.config.config import *




# Set up hardware params:
seed_test = RANDOM_STATE
random.seed(seed_test)
np.random.seed(seed_test)
torch.manual_seed(seed_test)
torch.cuda.manual_seed_all(seed_test)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataloader_train = DataLoader(train_dataset_tensor, sampler=RandomSampler(train_dataset_tensor),
#                               batch_size=TRAIN_BATCH_SIZE)
# dataloader_test = DataLoader(test_dataset_tensor, sampler=RandomSampler(test_dataset_tensor),
#                              batch_size=TEST_BATCH_SIZE)


#######################################################################################################################
#####################################  Define some evaluation functions  ##############################################
#######################################################################################################################
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def model_evaluate_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    total_true = 0

    # Define return type:
    model_evaluate_metrics_result = {
        'precision_total': 0,
        'true_predict_total': 0,
        'data_size': 0,
        'classes': [],
        'precision_per_class': [],
        'true_predicts': [],
        'total_predicts': [],
        'recall': 0,
        'fscore': 0,
        'confusion_matrix': []
    }

    # Some metrics for each label:
    for label in np.unique(labels_flat):
        # Save labels:
        model_evaluate_metrics_result['classes'].append(label)
        # Num of trues and total preds:
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]

        model_evaluate_metrics_result['true_predicts'].append(len(y_preds[y_preds == label]))
        model_evaluate_metrics_result['total_predicts'].append(len(y_true))
        # Calcualte precision for each class:
        pred_accuracy = len(y_preds[y_preds == label]) / len(y_true)
        model_evaluate_metrics_result['precision_per_class'].append(pred_accuracy)

        # Accumulate total
        total_true += len(y_preds[y_preds == label])

        # Print result:
        print(f'class: {label}')
        print(f'accuracy: {len(y_preds[y_preds == label])}/{len(y_true)} = {pred_accuracy} \n')

    # Metrics for whole dataset:
    #precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted')
    conf_mat = confusion_matrix(labels_flat, preds_flat)
    model_evaluate_metrics_result['data_size'] = len(preds_flat)
    model_evaluate_metrics_result['precision_total'] = precision
    model_evaluate_metrics_result['true_predict_total'] = total_true
    model_evaluate_metrics_result['recall'] = recall
    model_evaluate_metrics_result['fscore'] = f1
    model_evaluate_metrics_result['confusion_matrix'] = conf_mat

    return model_evaluate_metrics_result


def display_evaluation_metrics(model_evaluate_metrics_result):
    print(str(model_evaluate_metrics_result))


#     print(f'precision_total: {model_evaluate_metrics_result['precision_total']}\n')
#     print(f'recall: {model_evaluate_metrics_result['recall']}\n')
#     print(f'fscore: {model_evaluate_metrics_result['fscore']}\n')
#     print(f'true_predict_total: {model_evaluate_metrics_result['true_predict_total']}\n')
#     print(f'data_size: {model_evaluate_metrics_result['data_size']}\n')
#     print(f'confusion_matrix: {model_evaluate_metrics_result['confusion_matrix']}\n')

#     for i in range(len(model_evaluate_metrics_result['classes'])):
#         print(f'Class: {model_evaluate_metrics_result['classes'][i]}\n')
#         print(f'Precision: {model_evaluate_metrics_result['true_predicts'][i]}/{model_evaluate_metrics_result['total_predicts'][i]} = {model_evaluate_metrics_result['precision_per_class'][i]}\n')


def evaluate(model, dataloader_test_param):
    model.eval()

    loss_test_total = 0
    predictions, true_tests = [], []

    for batch in dataloader_test_param:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        # Do not store gradient thus it runs faster
        #print(batch[0].is_cuda, batch[0].is_cuda)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_test_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_tests.append(label_ids)

    loss_test_avg = loss_test_total / len(dataloader_test_param)

    predictions = np.concatenate(predictions, axis=0)
    true_tests = np.concatenate(true_tests, axis=0)

    return loss_test_avg, predictions, true_tests


def compute_general_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    #precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted')

    acc = accuracy_score(labels_flat, preds_flat)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': preds_flat,
        'orginal_predictions': preds
    }


def confusion_matrix_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return confusion_matrix(labels_flat, preds_flat)

