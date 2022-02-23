#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  model_train.py
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
from bzrs_main.modules.ml_models.victim_finder.data_preprocess import *
from bzrs_main.modules.ml_models.victim_finder.model_eval import *
from bzrs_main.modules.ml_models.victim_finder.model_build import *



# Set up hardware params:
seed_test = RANDOM_STATE
random.seed(seed_test)
np.random.seed(seed_test)
torch.manual_seed(seed_test)
torch.cuda.manual_seed_all(seed_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(torch.__version__)



#######################################################################################################################
#####################################     Training our own model here    ##############################################
#######################################################################################################################
# Variables for storing evaluation results which will be used in data viz:
viz_data_dict_training = {
    'epochs': list(range(EPOCHS)),
    'train_loss': [],
    'train_recall': [],
    'train_fscore': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_recall': [],
    'test_fscore': [],
    'test_accuracy': [],
    'train_time': [],
    'pred_time': []
}

args = {
    '--cuda': 'use GPU',
    '--train': 'df_train.csv',
    '--dev': 'df_val.csv',
    '--test': 'df_test.csv',
    '--vocab': '../Embeddings/glove_word2id',
    '--embeddings': '../Embeddings/glove_embeddings.npy',
    '--seed': 0,
    '--batch-size': 4,
    '--hidden-size': 256,
    '--clip-grad': 5.0,
    '--log-every': 10,
    '--max-epoch': 10,
    '--patience': 5,
    '--max-num-trial': 5,
    '--lr-decay': 0.5,
    '--lr': 0.001,
    '--save-to': 'test.model',
    '--model-path': 'test.model',
    '--valid-niter': 500,
    '--dropout': 0.3,
    '--out-channel': 16,
    '--verbose': 'whether to output the test results'
}

# Setting pretrained model:
if MODEL_SELECTED == 1:
    model = BertForSequenceClassification.from_pretrained(
        NLP_MODEL,
        num_labels=NUM_OF_LABELS,
        output_attentions=False,
        output_hidden_states=False)
elif MODEL_SELECTED == 2:
    model = XLNetForSequenceClassification.from_pretrained(
        NLP_MODEL,
        num_labels=NUM_OF_LABELS,
        output_attentions=False,
        output_hidden_states=False)
elif MODEL_SELECTED == 3:
    model = AlbertForSequenceClassification.from_pretrained(
        NLP_MODEL,
        num_labels=NUM_OF_LABELS,
        output_attentions=False,
        output_hidden_states=False)
elif MODEL_SELECTED == 4:
    model = RobertaForSequenceClassification.from_pretrained(
        NLP_MODEL,
        num_labels=NUM_OF_LABELS,
        output_attentions=False,
        output_hidden_states=False)
elif MODEL_SELECTED == 5:
    model = DistilBertForSequenceClassification.from_pretrained(
        NLP_MODEL,
        num_labels=NUM_OF_LABELS,
        output_attentions=False,
        output_hidden_states=False)
elif MODEL_SELECTED == 6:
    model = NonlinearModel(args, device, 2, float(args['--dropout']))
elif MODEL_SELECTED == 7:
    model = CustomBertLSTMModel(args, device, float(args['--dropout']), 2, lstm_hidden_size=int(args['--hidden-size']))
elif MODEL_SELECTED == 8:
    model = CustomBertLSTMAttentionModel(args, device, float(args['--dropout']), 2, lstm_hidden_size=int(args['--hidden-size']))
elif MODEL_SELECTED == 9:
    model = CustomBertConvModel(args, device, float(args['--dropout']), 2, out_channel=int(args['--out-channel']))
elif MODEL_SELECTED == 10:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
elif MODEL_SELECTED == 11:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
elif MODEL_SELECTED == 12:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
else:
    print("Model not initialized!")

# Clear GPU Cache before training:
torch.cuda.empty_cache()


# lr which is learning rate should be 2e-5 -> 5e-5
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * EPOCHS)

model.to(device)
# print(device)

# Define the loss function:
cn_loss = torch.nn.CrossEntropyLoss()

start_time = time.time()
# This cell is used to train the model, you can skip this cell and used the trained model instead in the next cell
for epoch in tqdm(trange(1, EPOCHS + 1)):
    epoch_start_time = time.time()
    model.train()

    loss_train_total = 0
    progress_bar = tqdm(dataloader_train, desc='Epoch {}'.format(epoch), leave=False, disable=False)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]}

        # print(inputs)
        outputs = model(**inputs)

        loss = outputs[0]
        # print(loss)
        # print(loss.item())
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    epoch_end_time = time.time()

    torch.save(model.state_dict(), f'Models/{NLP_MODEL}_{LABEL_TYPE}_epoch{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    pred_start_time = time.time()

    test_loss, predictions, true_tests = evaluate(model, dataloader_test)

    pred_end_time = time.time()

    train_loss, train_predictions, true_trains = evaluate(model, dataloader_train)

    test_f1 = f1_score_func(predictions, true_tests)

    train_f1 = f1_score_func(train_predictions, true_trains)

    test_general_metrics = compute_general_metrics(predictions, true_tests)

    train_general_metrics = compute_general_metrics(train_predictions, true_trains)

    train_time = epoch_end_time - epoch_start_time
    pred_time = pred_end_time - pred_start_time
    tqdm.write(f'Test loss: {test_loss}')
    tqdm.write(f'F1 score weighted: {test_f1}')
    tqdm.write(f'Training Time: {train_time}')
    tqdm.write(f'Testing Time: {pred_time}')
    # tqdm.write(f'General Metrics: {general_metrics}')

    # Save metrics to list:
    viz_data_dict_training['train_loss'].append(loss_train_avg)
    viz_data_dict_training['test_loss'].append(test_loss)
    viz_data_dict_training['test_recall'].append(test_general_metrics['recall'])
    viz_data_dict_training['test_fscore'].append(test_general_metrics['f1'])
    viz_data_dict_training['test_accuracy'].append(test_general_metrics['accuracy'])
    viz_data_dict_training['train_recall'].append(train_general_metrics['recall'])
    viz_data_dict_training['train_fscore'].append(train_general_metrics['f1'])
    viz_data_dict_training['train_accuracy'].append(train_general_metrics['accuracy'])
    viz_data_dict_training['train_time'].append(train_time)
    viz_data_dict_training['pred_time'].append(pred_time)

print('Done! time elapsed %.2f sec' % (time.time() - start_time))

# Save training metrics to file:
# Don't run this cell while visualizing data!!!
# Pickle data to file:
with open(TRAIN_EVAL_FILENAME, 'wb') as file:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(viz_data_dict_training, file, pickle.HIGHEST_PROTOCOL)

# Test: Read pickled file:
with open(TRAIN_EVAL_FILENAME, 'rb') as file:
    # The protocol version used is detected automatically, so we do not have to specify it.
    tmp_data = pickle.load(file)
    print(tmp_data)


# Evaluate time
#print(tmp_data["train_time"])
train_time_avg = np.mean(tmp_data["train_time"])
pred_time_avg = np.mean(tmp_data["pred_time"])
print(TRAIN_EVAL_FILENAME)
print(f'TrainTimeAvg: {train_time_avg}, PredTimeAvg: {pred_time_avg}')