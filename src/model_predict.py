#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  model_predict.py
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


class ModelPrediction():
    def __init__(self, model_type=1, model_name=NLP_MODEL, label_type=LABEL_TYPE):
        self.model_name = model_name
        self.label_name = label_type
        # Initializing model and pretrained params:
        if model_type == 1:
            self.model = BertForSequenceClassification.from_pretrained(NLP_MODEL,
                                                                   num_labels=NUM_OF_LABELS,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)
        elif model_type == 2:
            self.model = XLNetForSequenceClassification.from_pretrained(
                NLP_MODEL,
                num_labels = NUM_OF_LABELS,
                output_attentions=False,
                output_hidden_states=False)
        elif model_type == 3:
            self.model = AlbertForSequenceClassification.from_pretrained(
                NLP_MODEL,
                num_labels = NUM_OF_LABELS,
                output_attentions=False,
                output_hidden_states=False)
        elif model_type == 4:
            self.model = RobertaForSequenceClassification.from_pretrained(
                NLP_MODEL,
                num_labels = NUM_OF_LABELS,
                output_attentions=False,
                output_hidden_states=False)
        elif model_type == 5:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                NLP_MODEL,
                num_labels = NUM_OF_LABELS,
                output_attentions=False,
                output_hidden_states=False)
        elif model_type == 6:
            self.model = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
        elif model_type == 7:
            self.model = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
        elif model_type == 8:
            self.model = BertTokenizer.from_pretrained(TOKENIZER_MODEL, do_lower_case=True)
        else:
            print("Tokenizer not initialized!")
            raise Exception
        # Push model to device
        self.model.to(device)
        # Initialize model param using the best model:
        load_result = self.model.load_state_dict(
            torch.load('E:\\CodingProjects\\GRITFramework\\BZResearchStack\\res\\models\\Models\\bert-base-uncased_help_epoch27.model', map_location=torch.device('cpu')))
        print(load_result)


    def predict_text(self, str_text):

        return 0


    def predict_array(self, np_arrary_data):
        return np.array(0)


    def predict_tagged_data(self, tagged_data):
        return evaluate(self.model, tagged_data)



# Main method for testing:
def main():
    test_text_1 = "Please rescue me, I've got stucked somewhere in the water!!!"
    test_text_2 = "The weather of today is extremely great! God bless it!"
    #test_text_2 = "The weather of today is extremely great! God bless it!"
    bert_predict = ModelPrediction()
    result1 = bert_predict.predict_text(test_text_1)
    result1_1 = bert_predict.predict_text(test_text_1)
    result1_2 = bert_predict.predict_text(test_text_1)
    result1_3 = bert_predict.predict_text(test_text_1)
    result1_4 = bert_predict.predict_text(test_text_1)
    result2 = bert_predict.predict_text(test_text_2)
    result2_1 = bert_predict.predict_text(test_text_2)
    _, predictions, true_tests = bert_predict.predict_tagged_data(dataloader_test)
    print(predictions)
    print(result1)
    print(result1_1)
    print(result1_2)
    print(result1_3)
    print(result1_4)
    print(result2)
    print(result2_1)
    return 0



if __name__ == '__main__':
    main()