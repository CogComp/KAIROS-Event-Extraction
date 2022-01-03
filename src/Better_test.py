import ujson as json
import json
import tokenizations
import re
import nltk
from nltk.tokenize import word_tokenize
import torch
import json
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import numpy as np
import argparse
import os
from tqdm import tqdm
import spacy
from multiprocessing import Pool
from torch.nn.modules.distance import PairwiseDistance
from gurobi import *
from scipy.special import softmax
from util import *


def print_performance():
    print("Predicate by coarse (3): {}, {}/{}".format(predicate_by_coarse_3_cnt / predicate_total_cnt,
                                                      predicate_by_coarse_3_cnt,
                                                      predicate_total_cnt))
    print("Predicate by coarse (5): {}, {}/{}".format(predicate_by_coarse_5_cnt / predicate_total_cnt,
                                                      predicate_by_coarse_5_cnt,
                                                      predicate_total_cnt))
    print("Predicate by coarse (10): {}, {}/{}".format(predicate_by_coarse_10_cnt / predicate_total_cnt,
                                                       predicate_by_coarse_10_cnt,
                                                       predicate_total_cnt))
    print("Predicate HIT@1: {}, {}/{}".format(predicate_top1_cnt / predicate_total_cnt, predicate_top1_cnt,
                                              predicate_total_cnt))
    print("Predicate HIT@3: {}, {}/{}".format(predicate_top3_cnt / predicate_total_cnt, predicate_top3_cnt,
                                              predicate_total_cnt))
    print("Predicate HIT@5: {}, {}/{}".format(predicate_top5_cnt / predicate_total_cnt, predicate_top5_cnt,
                                              predicate_total_cnt))

    # print("Argument HIT@1: {}, {}/{}".format(argument_top1_cnt / argument_total_cnt, argument_top1_cnt,
    #                                          argument_total_cnt))
    # print("Argument HIT@3: {}, {}/{}".format(argument_top3_cnt / argument_total_cnt, argument_top3_cnt,
    #                                          argument_total_cnt))
    # print("Argument HIT@5: {}, {}/{}".format(argument_top5_cnt / argument_total_cnt, argument_top5_cnt,
    #                                          argument_total_cnt))


def read_data_from_bpjson(file_path):
    with open(file_path, 'r') as f:
        entries = json.load(f)["entries"]
        outputs = []
        for doc_id in entries.keys():
            events = entries[doc_id]["annotation-sets"]["basic-events"]["events"]  # event dict
            span_sets = entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"]  # span-sets dict
            segment_text = entries[doc_id]["segment-text"]
            # tokens=nltk.word_tokenize(segment_text)  # get tokens from article
            tokens = re.findall(r"\w+|[^\w\s]", segment_text, re.UNICODE)  # tokenize an article
            token_spans = tokenizations.get_original_spans(tokens, segment_text)  # get their spans
            # print(token_spans)
            global tags
            tags = ['O'] * len(tokens)
            span_dict = {}
            for span_id in span_sets.keys():
                span_dict[span_id] = [(span["start"], span["end"]) for span in span_sets[span_id]["spans"]]
            for event_id in events.keys():
                event_type = events[event_id]["event-type"]
                anchor_spans = span_dict[events[event_id]["anchors"]]
                update_tags(anchor_spans, token_spans, event_type)
            outputs.append({"tokens": tokens, "tags": tags})
    # print(outputs[-1])
    return outputs  # outputs format(per article): {"tokens":['t1','t2',...],"tags":['O','B-XXX',...]}


def update_tags(anchor_spans, token_spans, event_type):
    for s_anchor, e_anchor in anchor_spans:  # list of tuples
        s_index = -1
        e_index = -1
        for index in range(len(token_spans)):  # list of tuples
            s_token = token_spans[index][0]
            e_token = token_spans[index][1]
            if s_anchor == s_token:
                s_index = index
            if e_anchor == e_token:
                e_index = index
            if all([s_index != -1, e_index != -1]):
                break
        if any([s_index == -1, e_index == -1]):
            print(s_anchor, e_anchor)
            print(token_spans)
            print("didn't find any match!")
        else:
            tags[s_index] = 'B-' + event_type
            for i in range(e_index - s_index):
                tags[s_index + i + 1] = 'I-' + event_type


def convert_data_to_bert(file_path, data):
    with open(file_path, 'w') as f:
        for event_dict in data:
            for i in range(len(event_dict["tokens"])):
                f.write(event_dict["tokens"][i] + ' ' + event_dict["tags"][i] + '\n')
            f.write('\n')


# not include all the types
def get_labels_from_bpjson(input_file_path, output_file_path):
    labels = set()
    with open(input_file_path, 'r') as f:
        entries = json.load(f)["entries"]
        for doc_id in entries.keys():
            events = entries[doc_id]["annotation-sets"]["basic-events"]["events"]
            for event_id in events.keys():
                event_type = events[event_id]["event-type"]
                labels.add(event_type)
    with open(output_file_path, 'w') as f:
        for event_type in labels:
            f.write('B-' + event_type + '\n')
            f.write('I-' + event_type + '\n')
        f.write('O')


def get_labels(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f_r:
        with open(output_file_path, 'w') as f_w:
            while True:
                line = f_r.readline()
                if not line:
                    break
                else:
                    f_w.write('B-' + line)
                    f_w.write('I-' + line)
            f_w.write('O')


train_file_name = 'better_data/basic.eng-provided-72.0pct.train-70.0pct.d.bp.json'
test_file_name = 'better_data/basic.eng-provided-72.0pct.devtest-15.0pct.ref.d.bp-1.json'
arabic_file_name = 'better_data/basic_arabic_sample.d.bp.json'
additional_test_file_name = 'better_data/basic.eng-provided-72.0pct.analysis-15.0pct.ref.d.bp.json'


def load_trigger_examples(file_name):
    trigger_examples = list()
    with open(file_name, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        entries = train_data["entries"]
        outputs = []
        for doc_id in entries.keys():
            events = entries[doc_id]["annotation-sets"]["basic-events"]["events"]  # event dict
            span_sets = entries[doc_id]["annotation-sets"]["basic-events"]["span-sets"]  # span-sets dict
            segment_text = entries[doc_id]["segment-text"]
            # tokens=nltk.word_tokenize(segment_text)  # get tokens from article
            tokens = re.findall(r"\w+|[^\w\s]", segment_text, re.UNICODE)  # tokenize an article
            token_spans = tokenizations.get_original_spans(tokens, segment_text)

            # tokens_
            # get their spans
            # print(token_spans)
            global tags
            tags = ['O'] * len(tokens)
            span_dict = {}
            for span_id in span_sets.keys():
                span_dict[span_id] = [(span["start"], span["end"]) for span in span_sets[span_id]["spans"]]
            for event_id in events.keys():
                event_type = events[event_id]["event-type"]
                anchor_spans = span_dict[events[event_id]["anchors"]]
                for i, tmp_token_span in enumerate(token_spans):
                    if tmp_token_span in anchor_spans:
                        tags[i] = event_type
                        break

            # we then need to split the
            start_loc = 0
            tmp_training_examples = list()
            for i, token in enumerate(tokens):
                if token == '.':
                    for j in range(start_loc, i + 1):
                        if tags[j] != 'O':
                            tmp_training_examples.append(
                                {'tokens': tokens[start_loc: i + 1], 'trigger_pos': j - start_loc,
                                 'event_type': tags[j]})
                    start_loc = i + 1
            for j in range(start_loc, len(tokens)):
                if tags[j] != 'O':
                    tmp_training_examples.append(
                        {'tokens': tokens[start_loc:], 'trigger_pos': j - start_loc, 'event_type': tags[j]})
            trigger_examples += tmp_training_examples
        return trigger_examples


Better_event_types = ['Communicate-Event', 'Coordinated-Comm', 'Conduct-Protest', 'Conduct-Violent-Protest',
                      'Organize-Protest', 'Suppress-Communication', 'Suppress-Meeting', 'Suppress-or-Breakup-Protest',
                      'Corruption', 'Bribery', 'Extortion', 'Financial-Crime', 'Violence', 'Conspiracy', 'Coup',
                      'Suppression-of-Free-Speech', 'Persecution', 'Illegal-Entry', 'Other-Crime', 'Violence-Attack',
                      'Violence-Bombing', 'Violence-Set-Fire', 'Violence-Kill', 'Violence-Wound', 'Violence-Damage',
                      'Violence-Other', 'Change-of-Govt', 'Military-Declare-War', 'Military-Attack', 'Military-Other',
                      'Political-Election-Event', 'Political-Other', 'Fiscal-or-Monetary-Action',
                      'Other-Government-Action', 'Kidnapping', 'Law-Enforcement-Investigate',
                      'Law-Enforcement-Arrest', 'Law-Enforcement-Extradite', 'Law-Enforcement-Other', 'Judicial-Indict',
                      'Judicial-Prosecute', 'Judicial-Convict', 'Judicial-Sentence', 'Judicial-Acquit',
                      'Judicial-Seize',
                      'Judicial-Plead', 'Judicial-Other', 'Conduct-Meeting', 'Leave-Job', 'Economic-Event-or-SoA',
                      'Environmental-Event-or-SoA', 'Business-Event-or-SoA', 'Political-Event-or-SoA',
                      'War-Event-or-SoA',
                      'Famine-Event-or-SoA', 'Declare-Emergency', 'Monitor-Disease', 'Restrict-Travel',
                      'Impose-Quarantine', 'Close-Schools', 'Require-PPE', 'Restrict-Business', 'Cull-Livestock',
                      'Apply-NPI', 'Hospitalize', 'Vaccinate', 'Test-Patient', 'Disease-Outbreak', 'Disease-Infects',
                      'Disease-Exposes', 'Disease-Kills', 'Disease-Recovery']

Better_hierachical_structure = dict()
Better_hierachical_structure['Communicate'] = ['Communicate-Event', 'Coordinated-Comm']
Better_hierachical_structure['Protest'] = ['Conduct-Protest', 'Conduct-Violent-Protest', 'Organize-Protest']
Better_hierachical_structure['Persecution'] = ['Suppress-Communication', 'Suppress-or-Breakup-Protest',
                                               'Suppression-of-Free-Speech', 'Persecution']
Better_hierachical_structure['Crime'] = ['Kidnapping', 'Corruption', 'Bribery', 'Extortion', 'Financial-Crime',
                                         'Illegal-Entry', 'Conspiracy', 'Coup', 'Other-Crime']
Better_hierachical_structure['Government_disease_control'] = ['Restrict-Travel', 'Monitor-Disease', 'Impose-Quarantine',
                                                              'Restrict-Business', 'Cull-Livestock', 'Require-PPE',
                                                              'Close-Schools', 'Test-Patient']
Better_hierachical_structure['General_Government_action'] = ['Declare-Emergency', 'Other-Government-Action']
Better_hierachical_structure['Environment'] = ['Environmental-Event-or-SoA']
Better_hierachical_structure['Violence'] = ['Violence', 'Violence-Attack', 'Violence-Bombing', 'Violence-Set-Fire',
                                            'Violence-Kill', 'Violence-Wound', 'Violence-Damage', 'Violence-Other']
Better_hierachical_structure['Judicial'] = ['Law-Enforcement-Investigate', 'Law-Enforcement-Arrest',
                                            'Law-Enforcement-Extradite', 'Law-Enforcement-Other', 'Judicial-Indict',
                                            'Judicial-Prosecute', 'Judicial-Convict', 'Judicial-Sentence',
                                            'Judicial-Acquit', 'Judicial-Seize', 'Judicial-Plead', 'Judicial-Other']
Better_hierachical_structure['Meeting'] = ['Conduct-Meeting', 'Suppress-Meeting']
Better_hierachical_structure['Disease'] = ['Disease-Outbreak', 'Disease-Infects', 'Disease-Exposes', 'Disease-Kills',
                                           'Disease-Recovery', 'Hospitalize', 'Vaccinate', 'Apply-NPI']
Better_hierachical_structure['Political'] = ['Political-Event-or-SoA', 'Change-of-Govt', 'Political-Election-Event',
                                             'Political-Other']
Better_hierachical_structure['Military'] = ['War-Event-or-SoA', 'Military-Declare-War', 'Military-Attack',
                                            'Military-Other']
Better_hierachical_structure['Economy'] = ['Fiscal-or-Monetary-Action', 'Leave-Job', 'Economic-Event-or-SoA',
                                           'Business-Event-or-SoA', 'Famine-Event-or-SoA']

fine_type_2_coarse_type = dict()
for tmp_e_type in Better_event_types:
    for tmp_coarse_type in Better_hierachical_structure:
        if tmp_e_type in Better_hierachical_structure[tmp_coarse_type]:
            fine_type_2_coarse_type[tmp_e_type] = tmp_coarse_type
            break

print(len(fine_type_2_coarse_type))


# Suppress-Meeting
# Military-Declare-War
# Famine-Event-or-SoA
# Require-PPE

def evaluate_with_coarse_types(predictions, gold_label, limit_range=5):
    coarse_type_count = dict()
    for tmp_type in predictions[:limit_range]:
        tmp_coarse_type = fine_type_2_coarse_type[tmp_type]
        if tmp_coarse_type not in coarse_type_count:
            coarse_type_count[tmp_coarse_type] = 0
        coarse_type_count[tmp_coarse_type] += 1
    print('gold label:', gold_label)
    print('original prediction', predictions[:limit_range])
    print('coarse type count', coarse_type_count)
    sorted_coarse_type = sorted(coarse_type_count, key=lambda x: coarse_type_count[x], reverse=True)
    print('sorted coarse types:', sorted_coarse_type)
    if coarse_type_count[sorted_coarse_type[0]] >= limit_range / 2:

        for tmp_type in predictions[:limit_range]:
            tmp_coarse_type = fine_type_2_coarse_type[tmp_type]
            if tmp_coarse_type == sorted_coarse_type[0]:
                if tmp_type == gold_label:
                    print('coarse true')
                    return True
                else:
                    print('coarse false')
                    return False
    else:
        if predictions[0] == gold_label:
            print('no coarse true')
            return True
        else:
            print('no coarse false')
            return False
    print('-' * 64)


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='1', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--representation_source", default='nyt', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='bert-large', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--pooling_method", default='final', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--weight", default=10, type=float, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--num_anchor", default=10, type=int, required=False,
                    help="weight assigned to triggers")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_examples = load_trigger_examples(train_file_name)
test_examples = load_trigger_examples(test_file_name)
arabic_test_data = load_trigger_examples(arabic_file_name)
additional_test_examples = load_trigger_examples(additional_test_file_name)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)

anchor_sentences = dict()
for tmp_example in training_examples:
    if tmp_example['event_type'] not in anchor_sentences:
        anchor_sentences[tmp_example['event_type']] = list()
    anchor_sentences[tmp_example['event_type']].append(tmp_example)

anchor_sentences['Suppress-Meeting'] = [
    {'tokens': ['We', 'are', 'having', 'a', 'meeting', '.'], 'trigger_pos': 4, 'event_type': 'Suppress-Meeting'}]
anchor_sentences['Military-Declare-War'] = [
    {'tokens': ['germany', 'declares', 'war', 'to', 'Poland', '.'], 'trigger_pos': 1,
     'event_type': 'Military-Declare-War'}]
anchor_sentences['Famine-Event-or-SoA'] = [
    {'tokens': ['There', 'is', 'a', 'huge', 'Famine', '.'], 'trigger_pos': 4, 'event_type': 'Famine-Event-or-SoA'}]
anchor_sentences['Require-PPE'] = [
    {'tokens': ['This', 'project', 'requires', 'PPE', '.'], 'trigger_pos': 2, 'event_type': 'Require-PPE'}]

model.eval()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-20)

etype_to_distinct_embeddings = dict()

for tmp_e_type in tqdm(anchor_sentences, desc='Loading predicate embeddings'):
    etype_to_distinct_embeddings[tmp_e_type] = list()
    for tmp_example in anchor_sentences[tmp_e_type]:
        sent_emb = get_represetation(tmp_example['tokens'],
                                     (tmp_example['trigger_pos'], tmp_example['trigger_pos'] + 1),
                                     tokenizer, model, device)
        if not isinstance(sent_emb, str):
            etype_to_distinct_embeddings[tmp_e_type].append(sent_emb)

predicate_top1_cnt = 0.0
predicate_top3_cnt = 0.0
predicate_top5_cnt = 0.0
predicate_total_cnt = 0.0

predicate_by_coarse_3_cnt = 0.0
predicate_by_coarse_5_cnt = 0.0
predicate_by_coarse_10_cnt = 0.0

# argument_top1_cnt = 0.0
# argument_top3_cnt = 0.0
# argument_top5_cnt = 0.0
# argument_total_cnt = 0.0
chosen_event_types = Better_event_types

performance_by_type = dict()

for tmp_example in tqdm(test_examples, desc="Evaluation : " + str(args.weight)):

    sent_emb = get_represetation(tmp_example['tokens'], (tmp_example['trigger_pos'], tmp_example['trigger_pos'] + 1),
                                 tokenizer, model, device)
    predicate_score = list()
    for tmp_etype in chosen_event_types:
        predicate_score.append(
            get_similarity_score(sent_emb, etype_to_distinct_embeddings[tmp_etype]))

    scores = list()
    for tmp_score in predicate_score:
        scores.append(torch.tensor(tmp_score).view(1, -1).to(device))
    sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
    sorted_etypes = list()
    if tmp_example['event_type'] not in performance_by_type:
        performance_by_type[tmp_example['event_type']] = dict()
        performance_by_type[tmp_example['event_type']]['correct'] = 0
        performance_by_type[tmp_example['event_type']]['all'] = 0
    performance_by_type[tmp_example['event_type']]['all'] += 1
    for tmp_position in argument_indexes.tolist()[0]:
        sorted_etypes.append(chosen_event_types[tmp_position])
    if tmp_example['event_type'] in sorted_etypes[:1]:
        predicate_top1_cnt += 1
        performance_by_type[tmp_example['event_type']]['correct'] += 1
    if tmp_example['event_type'] in sorted_etypes[:3]:
        predicate_top3_cnt += 1
    if tmp_example['event_type'] in sorted_etypes[:5]:
        predicate_top5_cnt += 1
    predicate_total_cnt += 1
    if evaluate_with_coarse_types(sorted_etypes, tmp_example['event_type'], 3):
        predicate_by_coarse_3_cnt += 1
    if evaluate_with_coarse_types(sorted_etypes, tmp_example['event_type'], 5):
        predicate_by_coarse_5_cnt += 1
    if evaluate_with_coarse_types(sorted_etypes, tmp_example['event_type'], 10):
        predicate_by_coarse_10_cnt += 1

    # print_performance()

print_performance()
# print(candidate_length / total_cnt)

for tmp_e_type in performance_by_type:
    print(tmp_e_type, ':', performance_by_type[tmp_e_type]['correct'], '/', performance_by_type[tmp_e_type]['all'], performance_by_type[tmp_e_type]['correct']/performance_by_type[tmp_e_type]['all'])

print('end')
