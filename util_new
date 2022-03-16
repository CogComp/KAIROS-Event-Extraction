import spacy
import shutil
import ujson as json
import os
from tqdm import tqdm
from multiprocessing import Pool
import pandas
import torch
import argparse
from allennlp.predictors.predictor import Predictor
from transformers import BertTokenizer, BertModel
import allennlp_models.tagging
from gurobi import *
import numpy as np
from spacy.lemmatizer import Lemmatizer
import requests
from time import time

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
# from spacy.lookups import Lookups


stop_words_list = set(stopwords.words('english'))
exclude_words_list = set({ "according", "suggest", "suggests", "suggested", "suggesting", "tell", "tells", "telling", "told", "say", "says", "said", "saying", "based", "would", "including", "understand", "understands", "understood", "think", "thinks", "thinking", "thought"})
def filter_words(w, exclude_words=(), no_stop_words=True, tag_prefix="", selected_tags=[], stop_words=stop_words_list):
    if w in exclude_words:
        # print(w, " in ", "exclude_words_list")
        return False
    if no_stop_words and w in stop_words:
        # print(w , " in ", stop_words)
        return False

    tagged = nltk.pos_tag([w])
    
    if (tag_prefix and tagged[0][1][:len(tag_prefix)]==tag_prefix) or (tagged[0][1] in selected_tags):
        return True
    # else:
    #     print(w, " is ", tagged)
    return True

def load_event_ontology(path):
    # we need to load the event ontology
    event_definitions = pandas.read_excel(path, sheet_name='events')
    event_schema_to_roles = dict()
    event_types = list()
    role_types = list()
    role_to_type = dict()
    trigger_keywords = dict()
    role_keywords = dict()
    for index, row in event_definitions.iterrows():
        event_name = row['Type'] + ':' + row['Subtype'] + ':' + row['Sub-subtype']
        event_types.append(event_name)
        trigger_keywords[event_name] = row['Keyword'].split('$$')
        roles = list()
        for i in range(6):
            tmp_key = 'arg' + str(i + 1) + ' label'
            tmp_keywords = 'arg' + str(i + 1) + ' keyword'
            tmp_constraint_key = 'arg' + str(i + 1) + ' type constraints'
            if isinstance(row[tmp_key], str):
                roles.append(row[tmp_key])
                role_types.append(row[tmp_key])
                raw_entity_types = row[tmp_constraint_key].split(', ')
                role_to_type[row[tmp_key]] = list()
                for tmp_entity_type in raw_entity_types:
                    if tmp_entity_type[-1] == ' ':
                        role_to_type[row[tmp_key]].append(tmp_entity_type[:-1])
                    else:
                        role_to_type[row[tmp_key]].append(tmp_entity_type)
                role_to_type[row[tmp_key]].append('NAN')
                role_keywords[row[tmp_key]] = row[tmp_keywords].split('$$')
        event_schema_to_roles[event_name] = roles
    role_types = list(set(role_types))

    return {'event_types': event_types, 'role_types': role_types, 'event_schema_to_roles': event_schema_to_roles,
            'role_to_type': role_to_type, 'trigger_keywords': trigger_keywords, 'role_keywords': role_keywords}


class gurobi_opt:
    def __init__(self, predicate_score, argument_scores, entity_types, selected_event_types, selected_role_types,
                 weight=10,
                 prediction_length=5):
        self.num_predicate_labels = len(predicate_score)
        self.num_argument_labels = len(argument_scores[0])
        self.num_max = max(self.num_predicate_labels, self.num_argument_labels)
        self.initial_predicate_score = predicate_score + [0] * (self.num_max - self.num_predicate_labels)  # 1*num_max
        # self.initial_predicate_score = list(softmax(np.asarray(predicate_score + [0] * (self.num_max - self.num_predicate_labels))))
        self.initial_argument_scores = list()
        for tmp_arg_pos in range(len(argument_scores)):
            self.initial_argument_scores.append(
                argument_scores[tmp_arg_pos] + [0] * (self.num_max - self.num_argument_labels))  # n*num_max
            # self.initial_argument_scores.append(list(softmax(np.asarray(argument_scores[tmp_arg_pos] + [0] * (self.num_max - self.num_argument_labels)))))
        self.prediction_length = prediction_length
        self.selected_event_types = selected_event_types
        self.selected_role_types = selected_role_types
        self.entity_types = entity_types
        self.weight = weight

    def optimize_all(self):
        predicate_prediction = list()
        argument_predictions = list()
        for _ in range(len(self.initial_argument_scores)):
            argument_predictions.append(list())
        # make prediction for predicates
        for prediction_iteration in range(self.prediction_length):
            # print('We need to update the scores')
            tmp_predicate_score = list()
            for e_type_id, tmp_score in enumerate(self.initial_predicate_score):
                if e_type_id < self.num_predicate_labels and self.selected_event_types[
                    e_type_id] in predicate_prediction:
                    tmp_predicate_score.append(0)
                else:
                    tmp_predicate_score.append(tmp_score)
            input_scores = [np.asarray(tmp_predicate_score)]
            for tmp_arg_pos in range(len(self.initial_argument_scores)):
                tmp_argument_score = list()
                for r_type_id, tmp_score in enumerate(self.initial_argument_scores[tmp_arg_pos]):
                    tmp_argument_score.append(tmp_score)
                input_scores.append(np.asarray(tmp_argument_score))
            input_scores_array = np.asarray(input_scores)
            optimized_scores = self.optimize(input_scores_array)
            for tmp_pos in range(len(optimized_scores)):
                if tmp_pos == 0:
                    # predicate
                    for k in range(self.num_max):
                        if optimized_scores[tmp_pos][k] > 0:
                            if k > self.num_predicate_labels - 1:
                                predicate_prediction.append('None')
                            else:
                                predicate_prediction.append(self.selected_event_types[k])
                            break
                else:
                    continue
        # make prediction for arguments
        for prediction_iteration in range(self.prediction_length):
            # print('We need to update the scores')
            tmp_predicate_score = list()
            for e_type_id, tmp_score in enumerate(self.initial_predicate_score):
                tmp_predicate_score.append(tmp_score)
            input_scores = [np.asarray(tmp_predicate_score)]
            for tmp_arg_pos in range(len(self.initial_argument_scores)):
                tmp_argument_score = list()
                for r_type_id, tmp_score in enumerate(self.initial_argument_scores[tmp_arg_pos]):
                    if r_type_id < self.num_argument_labels and self.selected_role_types[r_type_id] in \
                            argument_predictions[tmp_arg_pos]:
                        tmp_argument_score.append(0)
                    else:
                        tmp_argument_score.append(tmp_score)
                input_scores.append(np.asarray(tmp_argument_score))
            input_scores_array = np.asarray(input_scores)
            optimized_scores = self.optimize(input_scores_array)
            for tmp_pos in range(len(optimized_scores)):
                if tmp_pos == 0:
                    # predicate
                    continue
                else:
                    # arguments
                    for k in range(self.num_max):
                        if optimized_scores[tmp_pos][k] > 0:
                            # argument_predictions[tmp_pos - 1].append(self.selected_role_types[k])
                            if k > self.num_argument_labels - 1:
                                argument_predictions[tmp_pos - 1].append('None')
                            else:
                                argument_predictions[tmp_pos - 1].append(self.selected_role_types[k])
                            break
        return predicate_prediction, argument_predictions

    def optimize(self, input_scores):
        self.model = Model('lp')
        self.model.setParam('OutputFlag', False)
        self.x = self.model.addVars(input_scores.shape[0], self.num_max, lb=0.0, ub=1.0, obj=input_scores,
                                    vtype=GRB.INTEGER,
                                    name="x")
        # For each prediction, we can only predict one result
        self.model.addConstrs((self.sum_prob(i) == 1.0 for i in range(input_scores.shape[0])), name='prob_constrs')
        # For each event, we cannot assign multiple arguments to the same role
        self.model.addConstrs((self.pred_prob(input_scores.shape[0] - 1, k) <= 1.0 for k in range(self.num_max)),
                              name='no_repeat_prediction')
        # Add constraints from the definitions
        for tmp_e_position, tmp_e_type in enumerate(self.selected_event_types):
            for tmp_r_position, tmp_r_type in enumerate(self.selected_role_types):
                if tmp_r_type not in event_schema_to_roles[tmp_e_type]:
                    self.model.addConstrs(
                        (self.x[0, tmp_e_position] + self.x[tmp_row + 1, tmp_r_position] <= 1.0 for tmp_row in
                         range(input_scores.shape[0] - 1)), name='event_definition')
        #
        # add constraints from the entity types
        # for tmp_argument_pos in range(input_scores.shape[0] - 1):
        #     for tmp_r_position, tmp_r_type in enumerate(self.selected_role_types):
        #         if self.entity_types[tmp_argument_pos] not in role_to_type[tmp_r_type]:
        #             self.model.addConstr(self.x[tmp_argument_pos + 1, tmp_r_position] == 0.0)
        #
        # for tmp_e_position, tmp_e_type in enumerate(self.selected_event_types):
        #     possible_entitie_types = list()
        #     for tmp_r in event_schema_to_roles[tmp_e_type]:
        #         possible_entitie_types += role_to_type[tmp_r]
        #     for tmp_entity_type in self.entity_types:
        #         if tmp_entity_type not in possible_entitie_types:
        #             self.model.addConstr(self.x[0, tmp_e_position] == 0.0)
        #             break

        self.model.update()
        self.sum_score = 0.0
        for i in range(input_scores.shape[0]):
            if i == 0:
                for k in range(self.num_max):
                    self.sum_score += self.x[i, k] * input_scores[i][k] * self.weight * (input_scores.shape[0] - 1)
                    # self.sum_score += self.x[i, k] * input_scores[i][k] * 99 * (input_scores.shape[0] - 1)
            else:
                for k in range(self.num_max):
                    self.sum_score += self.x[i, k] * input_scores[i][k]
        self.model.setObjective(self.sum_score, GRB.MAXIMIZE)  # maximize profit
        self.model.optimize()
        # print(input_scores)
        results = list()
        for i in range(input_scores.shape[0]):
            tmp_scores = list()
            for k in range(self.num_max):
                tmp_scores.append(self.x[i, k].X)
            results.append(tmp_scores)
        # print(results)
        return results

    # self.model.printAttr('X')

    def __call__(self):
        for v in self.model.getVars():
            print('%s %g' % (v.varName, v.x))
        return self.model.getVars()

    def sum_prob(self, i):
        sum_prob = 0.0
        for k in range(self.num_max):
            sum_prob += self.x[i, k]
        return sum_prob

    def pred_prob(self, num_arguments, k):
        sum_prob = 0.0
        for arg_id in range(num_arguments):
            sum_prob += self.x[arg_id + 1, k]
        return sum_prob


def get_represetation(sentence, target_positions, tokenizer, model, device, representation_type='all'):
    start_position = target_positions[0]
    end_position = target_positions[1]
    tokens = list()
    masks = list()
    # new_tokens = tokenizer.encode('<s>', add_special_tokens=False)
    new_tokens = tokenizer.encode('[CLS]', add_special_tokens=False)
    tokens += new_tokens
    masks += [1] * len(new_tokens)
    token_start_position = 0
    token_end_position = -1

    for i, w in enumerate(sentence):
        if i == start_position:
            token_start_position = len(tokens)
        if i == end_position:
            token_end_position = len(tokens)
        if start_position <= i < end_position:
            if representation_type == 'all':
                new_tokens = tokenizer.encode(w, add_special_tokens=False)
            else:
                new_tokens = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
            tokens += new_tokens
            masks += [0] * len(new_tokens)
        else:
            new_tokens = tokenizer.encode(w, add_special_tokens=False)
            tokens += new_tokens
            masks += [1] * len(new_tokens)
    # new_tokens = tokenizer.encode('</s>', add_special_tokens=False)
    new_tokens = tokenizer.encode('[SEP]', add_special_tokens=False)
    tokens += new_tokens
    masks += [1] * len(new_tokens)
    if len(tokens) > 512:
        return 'Too long'
    tensorized_token = torch.tensor([tokens]).to(device)
    tensorized_mask = torch.tensor([masks]).to(device)
    if representation_type == 'all':
        resulted_embedding = torch.mean(model(tensorized_token)[0][:, token_start_position:token_end_position, :],
                                        dim=1)
    else:
        resulted_embedding = torch.mean(
            model(tensorized_token, attention_mask=tensorized_mask)[0][:, token_start_position:token_end_position, :],
            dim=1)
    return torch.tensor(resulted_embedding.tolist()).view(1, -1).to(device)


def make_prediction_new(sentence_emb, all_label_embs_list, all_event_types, etype_radius):
    sentence_embedding_pile = sentence_emb.repeat(len(all_label_embs_list), 1)
    all_label_embs = torch.cat(all_label_embs_list, dim=0)

    similarities = cos(sentence_embedding_pile, all_label_embs).view(1, -1)
    sorted_similarities, argument_indexes = torch.sort(similarities, dim=1, descending=True)
    # print(all_event_types[argument_indexes.tolist()[0][0]])
    sorted_event_types = list()
    for local_p in argument_indexes.tolist()[0]:
        sorted_event_types.append(all_event_types[local_p])

    # if sorted_event_types[0] in event_types and (1 - sorted_similarities.tolist()[0][0]) <= etype_radius[
    #     sorted_event_types[0]]:
    if sorted_event_types[0] in event_types:
        # print((1-sorted_similarities.tolist()[0][0]))
        return True, sorted_event_types
    else:
        return False, sorted_event_types


def get_similarity_score(sentence_emb, label_embs):
    sentence_emb = sentence_emb.view(1, -1)  # n*1024
    mean_embedding = torch.mean(torch.cat(label_embs, dim=0), dim=0).view(1, -1)  # n*1024

    similarities = cos(sentence_emb, mean_embedding).view(1, -1)  # 1*n
    return similarities.tolist()[0][0]  # [1,1]


def map_tokens_to_chars(tokens, sentence, previous_token):
    token_span = []
    pointer = 0
    for token in tokens:
        while True:
            if token[0] == sentence[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start + previous_token, end + previous_token])
    return token_span


def map_tokens_to_tokens(tokens, target_tokens):
    token_span = []
    next_pointer = 0
    for token in tokens:
        pointer = next_pointer
        while True:
            if token == target_tokens[pointer]:
                token_span.append(pointer)
                next_pointer = pointer + 1
                break
            else:
                pointer += 1
                if pointer >= len(target_tokens):
                    token_span.append(-1)
                    break
    return token_span


class CogcompKairosEventExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertModel.from_pretrained('bert-large-uncased').to(device)
        self.model.eval()

        with open('data/selected_e_types.json', 'r') as f:
            all_event_types = json.load(f)

        with open('data/selected_r_types.json', 'r') as f:
            all_role_types = json.load(f)

        self.all_trigger_keywords = list()
        for tmp_key_word in trigger_keywords:
            self.all_trigger_keywords += trigger_keywords[tmp_key_word]
        all_trigger_keywords = list(set(self.all_trigger_keywords))

        selected_all_event_types = list()
        with open('all_types/event.ontology.new.txt', 'r',
                  encoding='utf-8') as f:
            for line in f:
                words = line[:-1].split('\t')
                tmp_event_type = words[1] + ':' + words[2]
                tmp_keywords = words[1].split(' ')
                if tmp_event_type not in all_event_types:
                    continue
                no_overlap = True
                for tmp_k in tmp_keywords:
                    if tmp_k in all_trigger_keywords:
                        no_overlap = False
                        break
                if no_overlap:
                    selected_all_event_types.append(tmp_event_type)
        self.all_event_types = event_types + selected_all_event_types
        all_role_types = list(set(role_types + all_role_types))

        self.etype_to_distinct_embeddings = dict()
        self.rtype_to_distinct_embeddings = dict()

        self.etype_embeddings = list()
        self.rtype_embeddings = list()

        with open('data/etype_to_distinct_embeddings.json', 'r') as f:
            raw_etype_to_distinct_embeddings = json.load(f)

            for tmp_e_type in self.all_event_types:
                self.etype_to_distinct_embeddings[tmp_e_type] = list()
                for tmp_embedding in raw_etype_to_distinct_embeddings[tmp_e_type]:
                    self.etype_to_distinct_embeddings[tmp_e_type].append(
                        torch.tensor(tmp_embedding).view(1, -1).to(device))
                self.etype_embeddings.append(
                    torch.mean(torch.cat(self.etype_to_distinct_embeddings[tmp_e_type], dim=0), dim=0).view(1, -1))

        with open('data/rtype_to_distinct_embeddings.json', 'r') as f:
            raw_rtype_to_distinct_embeddings = json.load(f)
            for tmp_r_type in all_role_types:
                self.rtype_to_distinct_embeddings[tmp_r_type] = list()
                for tmp_embedding in raw_rtype_to_distinct_embeddings[tmp_r_type]:
                    self.rtype_to_distinct_embeddings[tmp_r_type].append(
                        torch.tensor(tmp_embedding).view(1, -1).to(device))
                self.rtype_embeddings.append(
                    torch.mean(torch.cat(self.rtype_to_distinct_embeddings[tmp_r_type], dim=0), dim=0).view(1, -1))

        self.etype_radius = dict()
        for tmp_e_type in event_types:
            mean_embedding = torch.mean(torch.cat(self.etype_to_distinct_embeddings[tmp_e_type], dim=0), dim=0).view(1,
                                                                                                                     -1)
            positive_embeddings = self.etype_to_distinct_embeddings[tmp_e_type]
            negative_embeddings = list()
            for tmp_other_e_type in self.all_event_types:
                if tmp_other_e_type != tmp_e_type:
                    negative_embeddings += self.etype_to_distinct_embeddings[tmp_other_e_type]
            positive_distance = torch.tensor(1.).to(device) - cos(torch.cat(positive_embeddings, dim=0),
                                                                  mean_embedding.repeat(len(positive_embeddings),
                                                                                        1)).view(1,
                                                                                                 -1)
            positive_distance = positive_distance.tolist()[0]
            negative_distance = torch.tensor(1.).to(device) - cos(torch.cat(negative_embeddings, dim=0),
                                                                  mean_embedding.repeat(len(negative_embeddings),
                                                                                        1)).view(1,
                                                                                                 -1)
            negative_distance = negative_distance.tolist()[0]
            best_radius = 0
            best_F1 = 0
            best_p = 0
            best_r = 0
            for tmp_radius in range(100):
                tmp_radius = tmp_radius / 100
                correct_count = 0
                predict_count = 0
                for tmp_positive_distance in positive_distance:
                    if tmp_positive_distance <= tmp_radius:
                        correct_count += 1
                        predict_count += 1
                for tmp_negative_distance in negative_distance:
                    if tmp_negative_distance <= tmp_radius:
                        predict_count += 1
                if correct_count == 0:
                    continue
                tmp_p = correct_count / predict_count
                tmp_r = correct_count / len(positive_distance)
                tmp_F1 = (2 * tmp_p * tmp_r) / (tmp_p + tmp_r)
                if tmp_F1 > best_F1:
                    best_F1 = tmp_F1
                    best_p = tmp_p
                    best_r = tmp_r
                    best_radius = tmp_radius
            self.etype_radius[tmp_e_type] = best_radius

    def extract(self, input_sentence):
        SRL_parsing_results = Verbal_SRL_parsing(input_sentence)
        # print(SRL_parsing_results)
        NER_parsing_results = NER_parsing(input_sentence)
        # print(NER_parsing_results)
        tmp_lemmas = list()
        words = sp(input_sentence)
        for w in words:
            tmp_lemmas.append(w.lemma_)

        predictions = list()

        identified_trigger_positions = list()
        detected_ners = dict()

        start_position = -1
        for tmp_position, tag in enumerate(NER_parsing_results['tags']):

            if 'U-' == tag[:2]:
                detected_ners[(tmp_position, tmp_position + 1)] = tag.split('-')[1].lower()
            if start_position >= 0:
                if tag[:2] not in ['I-', 'L-']:
                    detected_ners[(start_position, tmp_position)] = tag.split('-')[1].lower()
                    start_position = -1
            else:
                if 'B-' == tag[:2]:
                    start_position = start_position

        if start_position >= 0:
            detected_ners[(start_position, len(NER_parsing_results['tags']))] = \
                NER_parsing_results['tags'][-1].split('-')[
                    1].lower()

        trigger_to_arguments = dict()
        # print(SRL_parsing_results)
        for tmp_v in SRL_parsing_results['verbs']:
            if 'B-V' not in tmp_v['tags']:
                continue
            identified_trigger_positions.append((tmp_v['tags'].index('B-V'), tmp_v['tags'].index('B-V') + 1))

            tmp_identified_argument_positions = list()
            start_position = -1

            for tmp_position, tmp_tag in enumerate(tmp_v['tags']):
                if start_position >= 0:
                    if 'I-ARG' not in tmp_tag:
                        tmp_identified_argument_positions.append((start_position, tmp_position))
                        if 'B-ARG' in tmp_tag:
                            start_position = tmp_position
                        else:
                            start_position = -1
                else:
                    if tmp_tag in ['B-ARG0', 'B-ARG1', 'B-ARG2']:
                        start_position = tmp_position
            if start_position >= 0:
                tmp_identified_argument_positions.append((start_position, len(tmp_v['tags'])))
            argument_positions = list()
            for tmp_entity in detected_ners:
                for tmp_a in tmp_identified_argument_positions:
                    if tmp_entity[1] >= tmp_a[0] and tmp_entity[0] <= tmp_a[1]:
                        argument_positions.append(tmp_entity)
                        break
            trigger_to_arguments[
                (tmp_v['tags'].index('B-V'), tmp_v['tags'].index('B-V') + 1)] = tmp_identified_argument_positions

        for tmp_location, tmp_lemma in enumerate(tmp_lemmas):
            if (tmp_location, tmp_location + 1) not in identified_trigger_positions:
                if tmp_lemma in self.all_trigger_keywords:
                    identified_trigger_positions.append((tmp_location, tmp_location + 1))
                    trigger_to_arguments[(tmp_location, tmp_location + 1)] = list(detected_ners.keys())
        identified_trigger_positions = list(set(identified_trigger_positions))
        selected_trigger_positions = list()
        # print('identified trigger positions', identified_trigger_positions)
        for tmp_position in identified_trigger_positions:
            if tmp_lemmas[tmp_position[0]] in self.all_trigger_keywords:
                selected_trigger_positions.append(tmp_position)
                continue
            tmp_embedding = get_represetation(SRL_parsing_results['words'], (tmp_position[0], tmp_position[1]),
                                              self.tokenizer,
                                              self.model,
                                              self.device)
            decision, sorted_types = make_prediction_new(tmp_embedding, self.etype_embeddings, self.all_event_types,
                                                         self.etype_radius)
            if decision:
                selected_trigger_positions.append(tmp_position)
        # print('selected trigger positions', selected_trigger_positions)
        for tmp_trigger_position in selected_trigger_positions:
            tmp_embedding = get_represetation(SRL_parsing_results['words'],
                                              (tmp_trigger_position[0], tmp_trigger_position[1]),
                                              self.tokenizer, self.model,
                                              self.device)

            predicate_score = list()
            for tmp_etype in event_types:
                predicate_score.append(
                    get_similarity_score(tmp_embedding, self.etype_to_distinct_embeddings[tmp_etype]))
            argument_scores = list()
            entity_types = list()
            for tmp_argument_position in trigger_to_arguments[tmp_trigger_position]:
                sent_emb = get_represetation(NER_parsing_results['words'], (tmp_argument_position[0],
                                                                            tmp_argument_position[1]),
                                             self.tokenizer, self.model, self.device, representation_type='mask')
                tmp_scores = list()
                for tmp_r_type in role_types:
                    tmp_scores.append(get_similarity_score(sent_emb, self.rtype_to_distinct_embeddings[tmp_r_type]))
                argument_scores.append(tmp_scores)
                if tmp_argument_position in detected_ners:
                    entity_types.append(detected_ners[tmp_argument_position])
                else:
                    entity_types.append('NAN')
            if len(argument_scores) > 0:
                tmp_optimizer = gurobi_opt(predicate_score, argument_scores, entity_types, event_types,
                                           role_types, weight=10)

                optimized_predicates, optimized_arguments = tmp_optimizer.optimize_all()

                trigger_type_prediction = optimized_predicates[0]
                # if trigger_type_prediction == 'None':
                #     print(optimized_predicates)
                argument_type_predictions = list()
                for tmp_types in optimized_arguments:
                    argument_type_predictions.append(tmp_types[0])

            else:
                scores = list()
                for tmp_score in predicate_score:
                    scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
                sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
                sorted_etypes = list()
                for tmp_position in argument_indexes.tolist()[0]:
                    sorted_etypes.append(event_types[tmp_position])
                trigger_type_prediction = sorted_etypes[0]
                # if trigger_type_prediction == 'None':
                #     print(sorted_etypes)
                argument_type_predictions = list()
                for tmp_pos, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
                    scores = list()
                    for tmp_score in argument_scores[tmp_pos]:
                        scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
                    sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
                    sorted_rtypes = list()
                    for tmp_position in argument_indexes.tolist()[0]:
                        sorted_rtypes.append(role_types[tmp_position])
                    argument_type_predictions.append(sorted_rtypes[0])

            if trigger_type_prediction != 'None':
                tmp_event = dict()
                tmp_event['trigger'] = {'position': tmp_trigger_position, 'type': trigger_type_prediction}
                tmp_event['arguments'] = list()
                for i, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
                    tmp_event['arguments'].append(
                        {'position': tmp_argument_position, 'role': argument_type_predictions[i]})
                tmp_event['sentence'] = input_sentence
                tmp_event['tokens'] = NER_parsing_results['words']
                predictions.append(tmp_event)
        return predictions



def Get_CogComp_SRL_and_NER_results(input_sentence):
    # start_time = time()
    # Variables we need to fill
    tokens = list()
    identified_trigger_positions = list()
    detected_mentions = dict()
    trigger_to_arguments = dict()

    # We first work on Ruohao's entity detection system
    # print('Extracting the mentions.')
    headers = {'Content-type': 'application/json'}
    NER_response = requests.post('http://dickens.seas.upenn.edu:4022/ner/',
                                 json={"task": "kairos_ner", "text": input_sentence}, headers=headers)
    if NER_response.status_code != 200:
        return tokens, detected_mentions, identified_trigger_positions, trigger_to_arguments

    NER_result = json.loads(NER_response.text)
    tokens = NER_result['tokens']
    NER_selected_view = dict()
    for tmp_view in NER_result['views']:
        if tmp_view['viewName'] == 'NER_CONLL':
            NER_selected_view = tmp_view
    for tmp_detected_mention in NER_selected_view['viewData'][0]['constituents']:
        detected_mentions[(tmp_detected_mention['start'], tmp_detected_mention['end'])] = tmp_detected_mention['label']

    # We then work on Celine's SRL system.
    # print('Extracting the events.')
    # SRL_response = requests.get('http://dickens.seas.upenn.edu:4039/annotate', data=input_sentence)
    SRL_response = requests.post('http://leguin.seas.upenn.edu:4039/annotate',
                                 json={'sentence': input_sentence})
    if SRL_response.status_code != 200:
        return tokens, detected_mentions, identified_trigger_positions, trigger_to_arguments
    SRL_result = json.loads(SRL_response.text)
    SRL_tokens = SRL_result['tokens']
    SRL_sentences = SRL_result['sentences']
    # print('Match tokens.')
    token_mapping = map_tokens_to_tokens(SRL_tokens, tokens)

    # print(SRL_tokens)
    # print(tokens)
    # print(token_mapping)
    # print('Finish matching.')
    verb_SRL_view = dict()
    nominal_SRL_view = dict()
    for tmp_view in SRL_result['views']:
        if tmp_view['viewName'] == 'SRL_ONTONOTES':
            verb_SRL_view = tmp_view
        elif tmp_view['viewName'] == 'SRL_NOM_ALL':
            nominal_SRL_view = tmp_view

    # we first deal with the verb SRL.
    for tmp_mention in verb_SRL_view['viewData'][0]['constituents']:
        if 'properties' in tmp_mention:
            # this a trigger
            start_position = token_mapping[tmp_mention['start']]
            if tmp_mention['end'] < len(token_mapping):
                end_position = token_mapping[tmp_mention['end']]
            else:
                end_position = len(tokens)
            # print('We identified an event')
            # print(start_position)
            # print(end_position)
            if start_position >= 0 and end_position >= 0:
                identified_trigger_positions.append((start_position, end_position))
                if (start_position, end_position) not in trigger_to_arguments:
                    trigger_to_arguments[(start_position, end_position)] = list()

    for tmp_relation in verb_SRL_view['viewData'][0]['relations']:
        if tmp_relation['relationName'] in ['ARG0', 'ARG1', 'ARG2']:
            candidate_trigger = verb_SRL_view['viewData'][0]['constituents'][tmp_relation['srcConstituent']]
            candidate_argument = verb_SRL_view['viewData'][0]['constituents'][tmp_relation['targetConstituent']]
            trigger_start = token_mapping[candidate_trigger['start']]
            if candidate_trigger['end'] < len(token_mapping):
                trigger_end = token_mapping[candidate_trigger['end']]
            else:
                trigger_end = len(tokens)

            argument_start = token_mapping[candidate_argument['start']]
            if candidate_argument['end'] < len(token_mapping):
                argument_end = token_mapping[candidate_argument['end']]
            else:
                argument_end = len(tokens)
            # print(argument_start)
            # print(argument_end)
            if trigger_start >= 0 and trigger_end >= 0 and argument_start >= 0 and argument_end >= 0:
                # print('detected mentions:', detected_mentions)
                for tmp_mention in detected_mentions:
                    if tmp_mention[0] >= argument_start and tmp_mention[1] <= argument_end:
                        trigger_to_arguments[(trigger_start, trigger_end)].append(tmp_mention)

    # we then deal with the nominal SRL.
    for tmp_mention in nominal_SRL_view['viewData'][0]['constituents']:
        if 'properties' in tmp_mention:
            # this a trigger
            start_position = token_mapping[tmp_mention['start']]
            if tmp_mention['end'] < len(token_mapping):
                end_position = token_mapping[tmp_mention['end']]
            else:
                end_position = len(tokens)
            if start_position >= 0 and end_position >= 0:
                identified_trigger_positions.append((start_position, end_position))
                if (start_position, end_position) not in trigger_to_arguments:
                    trigger_to_arguments[(start_position, end_position)] = list()

    for tmp_relation in nominal_SRL_view['viewData'][0]['relations']:
        if tmp_relation['relationName'] in ['ARG0', 'ARG1', 'ARG2']:
            candidate_trigger = nominal_SRL_view['viewData'][0]['constituents'][tmp_relation['srcConstituent']]
            candidate_argument = nominal_SRL_view['viewData'][0]['constituents'][tmp_relation['targetConstituent']]
            trigger_start = token_mapping[candidate_trigger['start']]
            if candidate_trigger['end'] < len(token_mapping):
                trigger_end = token_mapping[candidate_trigger['end']]
            else:
                trigger_end = len(tokens)

            argument_start = token_mapping[candidate_argument['start']]
            if candidate_argument['end'] < len(token_mapping):
                argument_end = token_mapping[candidate_argument['end']]
            else:
                argument_end = len(tokens)
            if trigger_start > 0 and trigger_end > 0 and argument_start > 0 and argument_end > 0:
                for tmp_mention in detected_mentions:
                    if tmp_mention[1] <= argument_end and tmp_mention[0] >= argument_start:
                        trigger_to_arguments[(trigger_start, trigger_end)].append(tmp_mention)

    for tmp_trigger in trigger_to_arguments:
        trigger_to_arguments[tmp_trigger] = list(set(trigger_to_arguments[tmp_trigger]))

    # end_time = time()
    # print("***Processing Time (SRL & NER ) :", end_time - start_time)
    # print('tokens:', tokens)
    # print('detected mentions:', detected_mentions)
    # print('identified_trigger_positions:', identified_trigger_positions)
    # print('trigger_to_arguments:', trigger_to_arguments)

    return tokens, detected_mentions, identified_trigger_positions, trigger_to_arguments, verb_SRL_view, nominal_SRL_view, SRL_sentences


class CogcompKairosEventExtractorTest:
    def __init__(self, device, model):
        self.device = device
        if model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.model = BertModel.from_pretrained('bert-large-uncased').to(device)
        elif model == 'mbert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            self.model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
        self.model.eval()

        with open('data/selected_e_types.json', 'r') as f:
            all_event_types = json.load(f)

        with open('data/selected_r_types.json', 'r') as f:
            all_role_types = json.load(f)

        self.all_trigger_keywords = list()
        for tmp_key_word in trigger_keywords:
            self.all_trigger_keywords += trigger_keywords[tmp_key_word]
        all_trigger_keywords = list(set(self.all_trigger_keywords))

        selected_all_event_types = list()
        with open('all_types/event.ontology.new.txt', 'r',
                  encoding='utf-8') as f:
            for line in f:
                words = line[:-1].split('\t')
                tmp_event_type = words[1] + ':' + words[2]
                tmp_keywords = words[1].split(' ')
                if tmp_event_type not in all_event_types:
                    continue
                no_overlap = True
                for tmp_k in tmp_keywords:
                    if tmp_k in all_trigger_keywords:
                        no_overlap = False
                        break
                if no_overlap:
                    selected_all_event_types.append(tmp_event_type)
        self.all_event_types = event_types + selected_all_event_types
        all_role_types = list(set(role_types + all_role_types))

        self.etype_to_distinct_embeddings = dict()
        self.rtype_to_distinct_embeddings = dict()

        self.etype_embeddings = list()
        self.rtype_embeddings = list()

        with open('data/etype_to_distinct_embeddings.json', 'r') as f:
            raw_etype_to_distinct_embeddings = json.load(f)

            for tmp_e_type in self.all_event_types:
                self.etype_to_distinct_embeddings[tmp_e_type] = list()
                for tmp_embedding in raw_etype_to_distinct_embeddings[tmp_e_type]:
                    # print(len(tmp_embedding))
                    self.etype_to_distinct_embeddings[tmp_e_type].append(
                        torch.tensor(tmp_embedding).view(1, -1).to(device))
                self.etype_embeddings.append(
                    torch.mean(torch.cat(self.etype_to_distinct_embeddings[tmp_e_type], dim=0), dim=0).view(1, -1))

        with open('data/rtype_to_distinct_embeddings.json', 'r') as f:
            raw_rtype_to_distinct_embeddings = json.load(f)
            for tmp_r_type in all_role_types:
                self.rtype_to_distinct_embeddings[tmp_r_type] = list()
                for tmp_embedding in raw_rtype_to_distinct_embeddings[tmp_r_type]:
                    self.rtype_to_distinct_embeddings[tmp_r_type].append(
                        torch.tensor(tmp_embedding).view(1, -1).to(device))
                self.rtype_embeddings.append(
                    torch.mean(torch.cat(self.rtype_to_distinct_embeddings[tmp_r_type], dim=0), dim=0).view(1, -1))

        self.etype_radius = dict()
        for tmp_e_type in event_types:
            mean_embedding = torch.mean(torch.cat(self.etype_to_distinct_embeddings[tmp_e_type], dim=0), dim=0).view(1,
                                                                                                                     -1)
            positive_embeddings = self.etype_to_distinct_embeddings[tmp_e_type]
            negative_embeddings = list()
            for tmp_other_e_type in self.all_event_types:
                if tmp_other_e_type != tmp_e_type:
                    negative_embeddings += self.etype_to_distinct_embeddings[tmp_other_e_type]
            positive_distance = torch.tensor(1.).to(device) - cos(torch.cat(positive_embeddings, dim=0),
                                                                  mean_embedding.repeat(len(positive_embeddings),
                                                                                        1)).view(1,
                                                                                                 -1)
            positive_distance = positive_distance.tolist()[0]
            negative_distance = torch.tensor(1.).to(device) - cos(torch.cat(negative_embeddings, dim=0),
                                                                  mean_embedding.repeat(len(negative_embeddings),
                                                                                        1)).view(1,
                                                                                                 -1)
            negative_distance = negative_distance.tolist()[0]
            best_radius = 0
            best_F1 = 0
            best_p = 0
            best_r = 0
            for tmp_radius in range(100):
                tmp_radius = tmp_radius / 100
                correct_count = 0
                predict_count = 0
                for tmp_positive_distance in positive_distance:
                    if tmp_positive_distance <= tmp_radius:
                        correct_count += 1
                        predict_count += 1
                for tmp_negative_distance in negative_distance:
                    if tmp_negative_distance <= tmp_radius:
                        predict_count += 1
                if correct_count == 0:
                    continue
                tmp_p = correct_count / predict_count
                tmp_r = correct_count / len(positive_distance)
                tmp_F1 = (2 * tmp_p * tmp_r) / (tmp_p + tmp_r)
                if tmp_F1 > best_F1:
                    best_F1 = tmp_F1
                    best_p = tmp_p
                    best_r = tmp_r
                    best_radius = tmp_radius
            self.etype_radius[tmp_e_type] = best_radius

    def extract(self, input_sentence, include_all_verbs=False):
        tokens, detected_mentions, identified_trigger_positions, trigger_to_arguments, verb_SRL_view, nominal_SRL_view, SRL_sentences = Get_CogComp_SRL_and_NER_results(
            input_sentence)

        print("\n---TOKENS:")
        for i in range(len(tokens)):
            print(i, " : ", tokens[i], end=" , ")
        print("\n")

        print('---identified trigger positions : ', identified_trigger_positions)
        
        print("---Identified Triggers: \n")
        for i in range(len(identified_trigger_positions)):
            print(i, " : ", tokens[identified_trigger_positions[i][0]], end=" , ")
        print("\n")

        print('detected mentions:', detected_mentions)
        print('trigger_to_arguments:', trigger_to_arguments)
        # start_time = time()
        
        selected_trigger_positions = list()


        # start_time_onto = time()

        for tmp_position in identified_trigger_positions:
            tmp_embedding = get_represetation(tokens, (tmp_position[0], tmp_position[1]),
                                              self.tokenizer,
                                              self.model,
                                              self.device)
            decision, sorted_types = make_prediction_new(tmp_embedding, self.etype_embeddings, self.all_event_types,
                                                         self.etype_radius)
            # print(sorted_types[:5])
            if decision:
                selected_trigger_positions.append(tmp_position)
        # print("***Processing Time (Onto) : ", time() - start_time_onto)
        print('selected trigger positions : ', selected_trigger_positions)
        print("\n---Selected Triggers: \n")
        for i in range(len(selected_trigger_positions)):
            print(i, " : ", tokens[selected_trigger_positions[i][0]], end=" , ")
        print("\n")


        if include_all_verbs:
            selected_trigger_positions_set = set([(x,y) for (x,y) in selected_trigger_positions])
            # print("selected_trigger_positions_set: ", selected_trigger_positions_set)
            for data in verb_SRL_view['viewData']:
                for constituent in data['constituents']:
                    if constituent['label'] == 'Predicate':
                        tmp_pos = (constituent['start'] , constituent['end'])
                        if tmp_pos not in selected_trigger_positions_set:
                            if filter_words(tokens[tmp_pos[0]], exclude_words=exclude_words_list, tag_prefix="V", stop_words=stop_words_list): 
                                selected_trigger_positions.append(tmp_pos)

                print('selected trigger positions(all verbs included): ', selected_trigger_positions)
                print("\n---Selected Triggers((all verbs included)):")
                for i in range(len(selected_trigger_positions)):
                    print(i, " : ", tokens[selected_trigger_positions[i][0]], end=" , ")
                print("\n")

        # start_time_type = time()
        predictions = list()
        for tmp_trigger_position in selected_trigger_positions:
            # tmp_embedding = get_represetation(tokens,
            #                                   (tmp_trigger_position[0], tmp_trigger_position[1]),
            #                                   self.tokenizer, self.model,
            #                                   self.device)

            # predicate_score = list()
            # for tmp_etype in event_types:
            #     predicate_score.append(
            #         get_similarity_score(tmp_embedding, self.etype_to_distinct_embeddings[tmp_etype]))
            # argument_scores = list()
            # entity_types = list()
            # for tmp_argument_position in trigger_to_arguments[tmp_trigger_position]:
            #     sent_emb = get_represetation(tokens, (tmp_argument_position[0],
            #                                           tmp_argument_position[1]),
            #                                  self.tokenizer, self.model, self.device, representation_type='mask')
            #     tmp_scores = list()
            #     for tmp_r_type in role_types:
            #         tmp_scores.append(get_similarity_score(sent_emb, self.rtype_to_distinct_embeddings[tmp_r_type]))
            #     argument_scores.append(tmp_scores)
            #     if tmp_argument_position in detected_mentions:
            #         entity_types.append(detected_mentions[tmp_argument_position].lower())
            #     else:
            #         entity_types.append('NAN')
            # if len(argument_scores) > 0:
            #     tmp_optimizer = gurobi_opt(predicate_score, argument_scores, entity_types, event_types,
            #                                role_types, weight=10)

            #     optimized_predicates, optimized_arguments = tmp_optimizer.optimize_all()

            #     trigger_type_prediction = optimized_predicates[0]
            #     # if trigger_type_prediction == 'None':
            #     #     print(optimized_predicates)
            #     argument_type_predictions = list()
            #     for tmp_types in optimized_arguments:
            #         argument_type_predictions.append(tmp_types[0])

            # else:
            #     scores = list()
            #     for tmp_score in predicate_score:
            #         scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
            #     sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
            #     sorted_etypes = list()
            #     for tmp_position in argument_indexes.tolist()[0]:
            #         sorted_etypes.append(event_types[tmp_position])
            #     trigger_type_prediction = sorted_etypes[0]
            #     # if trigger_type_prediction == 'None':
            #     #     print(sorted_etypes)
            #     argument_type_predictions = list()
            #     for tmp_pos, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
            #         scores = list()
            #         for tmp_score in argument_scores[tmp_pos]:
            #             scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
            #         sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
            #         sorted_rtypes = list()
            #         for tmp_position in argument_indexes.tolist()[0]:
            #             sorted_rtypes.append(role_types[tmp_position])
            #         argument_type_predictions.append(sorted_rtypes[0])

            # if trigger_type_prediction != 'None':
            #     tmp_event = dict()
            #     tmp_event['trigger'] = {'position': tmp_trigger_position, 'type': trigger_type_prediction}
            #     tmp_event['arguments'] = list()
            #     for i, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
            #         tmp_event['arguments'].append(
            #             {'position': tmp_argument_position, 'role': argument_type_predictions[i],
            #              'entity_type': detected_mentions[tmp_argument_position].lower()})
            #     tmp_event['sentence'] = input_sentence
            #     tmp_event['tokens'] = tokens
            #     predictions.append(tmp_event)
            # else:
            #     tmp_event = dict()
            #     tmp_event['trigger'] = {'position': tmp_trigger_position, 'type': 'None'}
            #     tmp_event['arguments'] = list()
            #     for i, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
            #         tmp_event['arguments'].append(
            #             {'position': tmp_argument_position, 'role': argument_type_predictions[i],
            #              'entity_type': detected_mentions[tmp_argument_position].lower()})
            #     tmp_event['sentence'] = input_sentence
            #     tmp_event['tokens'] = tokens
            #     predictions.append(tmp_event)
            ### added
            tmp_event = dict()
            tmp_event['trigger'] = {'position': tmp_trigger_position, 'type': ''}
            tmp_event['arguments'] = list()
            for i, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
                tmp_event['arguments'].append(
                    {'position': tmp_argument_position, 'role': '',
                        'entity_type': detected_mentions[tmp_argument_position].lower()})
            tmp_event['sentence'] = input_sentence
            tmp_event['tokens'] = tokens
            predictions.append(tmp_event)
            #### end

        # print("***Processing Time(Typing) :", time()-start_time_type)
        # print("***Processing Time (extract except SRL and NER): ", time() - start_time)
        return predictions, tokens, SRL_sentences

    def extract_with_annotation(self, input_sentence, tokens, detected_mentions, identified_trigger_positions, trigger_to_arguments):

        selected_trigger_positions = list()
        print('identified trigger positions', identified_trigger_positions)
        for tmp_position in identified_trigger_positions:
            tmp_embedding = get_represetation(tokens, (tmp_position[0], tmp_position[1]),
                                              self.tokenizer,
                                              self.model,
                                              self.device)
            decision, sorted_types = make_prediction_new(tmp_embedding, self.etype_embeddings, self.all_event_types,
                                                         self.etype_radius)
            print(sorted_types[:5])
            if decision:
                selected_trigger_positions.append(tmp_position)
        print('selected trigger positions', selected_trigger_positions)

        predictions = list()
        for tmp_trigger_position in selected_trigger_positions:
            tmp_embedding = get_represetation(tokens,
                                              (tmp_trigger_position[0], tmp_trigger_position[1]),
                                              self.tokenizer, self.model,
                                              self.device)

            predicate_score = list()
            for tmp_etype in event_types:
                predicate_score.append(
                    get_similarity_score(tmp_embedding, self.etype_to_distinct_embeddings[tmp_etype]))
            argument_scores = list()
            entity_types = list()
            for tmp_argument_position in trigger_to_arguments[tmp_trigger_position]:
                sent_emb = get_represetation(tokens, (tmp_argument_position[0],
                                                      tmp_argument_position[1]),
                                             self.tokenizer, self.model, self.device, representation_type='mask')
                tmp_scores = list()
                for tmp_r_type in role_types:
                    tmp_scores.append(get_similarity_score(sent_emb, self.rtype_to_distinct_embeddings[tmp_r_type]))
                argument_scores.append(tmp_scores)
                if tmp_argument_position in detected_mentions:
                    entity_types.append(detected_mentions[tmp_argument_position].lower())
                else:
                    entity_types.append('NAN')
            if len(argument_scores) > 0:
                tmp_optimizer = gurobi_opt(predicate_score, argument_scores, entity_types, event_types,
                                           role_types, weight=10)

                optimized_predicates, optimized_arguments = tmp_optimizer.optimize_all()

                trigger_type_prediction = optimized_predicates[0]
                # if trigger_type_prediction == 'None':
                #     print(optimized_predicates)
                argument_type_predictions = list()
                for tmp_types in optimized_arguments:
                    argument_type_predictions.append(tmp_types[0])

            else:
                scores = list()
                for tmp_score in predicate_score:
                    scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
                sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
                sorted_etypes = list()
                for tmp_position in argument_indexes.tolist()[0]:
                    sorted_etypes.append(event_types[tmp_position])
                trigger_type_prediction = sorted_etypes[0]
                # if trigger_type_prediction == 'None':
                #     print(sorted_etypes)
                argument_type_predictions = list()
                for tmp_pos, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
                    scores = list()
                    for tmp_score in argument_scores[tmp_pos]:
                        scores.append(torch.tensor(tmp_score).view(1, -1).to(self.device))
                    sorted_similarities, argument_indexes = torch.sort(torch.cat(scores, dim=1), dim=1, descending=True)
                    sorted_rtypes = list()
                    for tmp_position in argument_indexes.tolist()[0]:
                        sorted_rtypes.append(role_types[tmp_position])
                    argument_type_predictions.append(sorted_rtypes[0])

            if trigger_type_prediction != 'None':
                tmp_event = dict()
                tmp_event['trigger'] = {'position': tmp_trigger_position, 'type': trigger_type_prediction}
                tmp_event['arguments'] = list()
                for i, tmp_argument_position in enumerate(trigger_to_arguments[tmp_trigger_position]):
                    tmp_event['arguments'].append(
                        {'position': tmp_argument_position, 'role': argument_type_predictions[i],
                         'entity_type': detected_mentions[tmp_argument_position].lower()})
                tmp_event['sentence'] = input_sentence
                tmp_event['tokens'] = tokens
                predictions.append(tmp_event)
        return predictions


# print('Start to load the SRL and NER models.')
# SRL_predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
# NER_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
# print('Finish loading the SRL and NER models.')


# def Verbal_SRL_parsing(input_sentence):
#     tmp_result = SRL_predictor.predict_batch_json([{'sentence': input_sentence}])[0]
#     return tmp_result
#
#
# def NER_parsing(input_sentence):
#     try:
#         tmp_result = NER_predictor.predict(sentence=input_sentence)
#         return tmp_result
#     except:
#         return {'logits': [], 'mask': [], 'tags': [], 'words': []}


# with open('data/ontology.json', 'r') as f:
#     ontology = json.load(f)
sp = spacy.load('en_core_web_sm')
ontology = load_event_ontology('data/KAIROS_Annotation_Tagset_Phase_1_V2.0.xlsx')
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-20)

event_types = ontology['event_types']
# print(event_types)
role_types = ontology['role_types']
event_schema_to_roles = ontology['event_schema_to_roles']
role_to_type = ontology['role_to_type']
trigger_keywords = ontology['trigger_keywords']
role_keywords = ontology['role_keywords']

with open('data/expan-v1.json', 'r') as f:
    expanded_trigger_keywords = json.load(f)

# for tmp_e_type in expanded_trigger_keywords:
#     for tmp_expanded_keyword in expanded_trigger_keywords[tmp_e_type]['expansion'][:3]:
#         trigger_keywords[tmp_e_type].append(tmp_expanded_keyword)
#     trigger_keywords[tmp_e_type] = list(set(trigger_keywords[tmp_e_type]))
