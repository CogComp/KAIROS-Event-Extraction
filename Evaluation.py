import ujson as json
import os
from util import *


# annotated_data_list = ['']
#
# with open('data/data_after_event_extraction.json', 'r') as f:
#     extracted_data = json.load(f)
#     print('lalala')


def load_annotation(file_path):
    annotation_results = dict()
    annotation_results['mentions'] = dict()
    annotation_results['events'] = list()
    annotation_results['relations'] = list()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line[:-1].split('\t')

            if words[0][0] == 'T':
                # This is an entity annotation
                tmp_key = words[0]
                tmp_type = words[1].split(' ')[0]
                start_pos = words[1].split(' ')[1]
                end_pos = words[1].split(' ')[2]
                text = words[2]
                annotation_results['mentions'][tmp_key] = {'start_pos': int(start_pos.split(';')[0]),
                                                           'end_pos': int(end_pos.split(';')[0]),
                                                           'type': tmp_type.replace('_', ':'),
                                                           'text': text}
            if words[0][0] == 'E':
                # print(words)
                # This is an event annotation
                tmp_event_key = words[0]
                event_type = words[1].split(' ')[0].split(':')[0]
                trigger = words[1].split(' ')[0].split(':')[1]
                arguments = list()
                for tmp_w in words[1].split(' ')[1:]:
                    try:
                        arguments.append({'role': tmp_w.split(':')[0], 'argument': tmp_w.split(':')[1]})
                    except:
                        continue
                # print({'event_type': event_type, 'trigger': trigger, 'arguments': arguments})
                annotation_results['events'].append(
                    {'key': tmp_event_key, 'event_type': event_type.replace('_', ':'), 'trigger': trigger,
                     'arguments': arguments})

            if words[0][0] == 'R':
                # print(words)
                # This is an event annotation
                tmp_relation_key = words[0]
                relation_type = words[1].split(' ')[0]
                head = words[1].split(' ')[1].split(':')[1]
                tail = words[1].split(' ')[2].split(':')[1]
                annotation_results['relations'].append(
                    {'key': tmp_relation_key, 'relation_type': relation_type, 'head': head, 'tail': tail})
    # print(annotation_results['mentions'])
    return annotation_results


def print_performance_by_type(result_by_type):
    for tmp_event_type in event_types:
        print('----' * 32)
        print('Performance on:', tmp_event_type)
        print('Number of gold triggers:', result_by_type[tmp_event_type]['num_trigger_gold'])
        print('Number of identified triggers:', result_by_type[tmp_event_type]['num_trigger_predict'])
        print('Number of gold arguments:', result_by_type[tmp_event_type]['num_argument_gold'])
        print('Number of identified arguments:', result_by_type[tmp_event_type]['num_argument_predict'])
        if 0 in [result_by_type[tmp_event_type]['num_trigger_gold'],
                 result_by_type[tmp_event_type]['num_trigger_predict'],
                 result_by_type[tmp_event_type]['num_argument_gold'],
                 result_by_type[tmp_event_type]['num_argument_predict']]:
            continue
        trigger_identification_p = result_by_type[tmp_event_type]['num_trigger_identification_correct'] / \
                                   result_by_type[tmp_event_type]['num_trigger_predict']
        trigger_identification_r = result_by_type[tmp_event_type]['num_trigger_identification_correct'] / \
                                   result_by_type[tmp_event_type]['num_trigger_gold']

        if trigger_identification_p > 0 and trigger_identification_r > 0:
            trigger_identification_f1 = (2 * trigger_identification_p * trigger_identification_r) / (
                    trigger_identification_p + trigger_identification_r)
        else:
            trigger_identification_f1 = 0
        print('Trigger identification:', trigger_identification_p, trigger_identification_r, trigger_identification_f1)

        trigger_classification_p = result_by_type[tmp_event_type]['num_trigger_classification_correct'] / \
                                   result_by_type[tmp_event_type]['num_trigger_predict']
        trigger_classification_r = result_by_type[tmp_event_type]['num_trigger_classification_correct'] / \
                                   result_by_type[tmp_event_type]['num_trigger_gold']
        if trigger_classification_p > 0 and trigger_classification_r > 0:
            trigger_classification_f1 = (2 * trigger_classification_p * trigger_classification_r) / (
                    trigger_classification_p + trigger_classification_r)
        else:
            trigger_classification_f1 = 0
        print('Trigger classification:', trigger_classification_p, trigger_classification_r, trigger_classification_f1)

        argument_identification_p = result_by_type[tmp_event_type]['num_argument_identification_correct'] / \
                                    result_by_type[tmp_event_type]['num_argument_predict']
        argument_identification_r = result_by_type[tmp_event_type]['num_argument_identification_correct'] / \
                                    result_by_type[tmp_event_type]['num_argument_gold']

        if argument_identification_p > 0 and argument_identification_r > 0:

            argument_identification_f1 = (2 * argument_identification_p * argument_identification_r) / (
                    argument_identification_p + argument_identification_r)
        else:
            argument_identification_f1 = 0
        print('Argument identification:', argument_identification_p, argument_identification_r,
              argument_identification_f1)

        argument_classification_p = result_by_type[tmp_event_type]['num_argument_classification_correct'] / \
                                    result_by_type[tmp_event_type]['num_argument_predict']
        argument_classification_r = result_by_type[tmp_event_type]['num_argument_classification_correct'] / \
                                    result_by_type[tmp_event_type]['num_argument_gold']

        if argument_classification_p > 0 and argument_classification_r > 0:
            argument_classification_f1 = (2 * argument_classification_p * argument_classification_r) / (
                    argument_classification_p + argument_classification_r)
        else:
            argument_classification_f1 = 0
        print('Trigger classification:', argument_classification_p, argument_classification_r,
              argument_classification_f1)


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


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='1', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--representation_source", default='nyt', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='bert-large', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--pooling_method", default='final', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--weight", default=100, type=float, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--argument_matching", default='exact', type=str, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--eval_model", default='joint', type=str, required=False,
                    help="weight assigned to triggers")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)

test_extractor = CogcompKairosEventExtractor(device)

# file_ids = list()
# file_names = os.listdir('data/Quizlet3_annotation')
# for tmp_file_name in file_names:
#     file_ids.append(tmp_file_name.split('.')[0])
#
# file_ids = list(set(file_ids))

num_trigger_predict = 0
num_trigger_identification_correct = 0
num_trigger_classification_correct = 0
num_trigger_gold = 0

num_argument_predict = 0
num_argument_identification_correct = 0
num_argument_classification_correct = 0
num_argument_gold = 0

result_by_type = dict()
for tmp_event_type in event_types:
    result_by_type[tmp_event_type] = dict()
    result_by_type[tmp_event_type]['num_trigger_predict'] = 0
    result_by_type[tmp_event_type]['num_trigger_identification_correct'] = 0
    result_by_type[tmp_event_type]['num_trigger_classification_correct'] = 0
    result_by_type[tmp_event_type]['num_trigger_gold'] = 0
    result_by_type[tmp_event_type]['num_argument_predict'] = 0
    result_by_type[tmp_event_type]['num_argument_identification_correct'] = 0
    result_by_type[tmp_event_type]['num_argument_classification_correct'] = 0
    result_by_type[tmp_event_type]['num_argument_gold'] = 0

dev_sets = list()

dev_sets.append('kairos_VOA/9_VOA_EN_NW_2016.09.19.3515911')
dev_sets.append('kairos_VOA/60_VOA_EN_NW_2017.06.06.3888881')
dev_sets.append('kairos_VOA/73_VOA_EN_NW_2016.01.19.3152458')
dev_sets.append('kairos_VOA/74_VOA_EN_NW_2017.05.24.3868599')

dev_sets.append('kairos_schema_sentences/89_damagedestroydisabledismantle_destroy')
dev_sets.append('kairos_schema_sentences/96_demonstrate_unspecified')
dev_sets.append('kairos_schema_sentences/749_requestcommand_correspondence')
dev_sets.append('kairos_schema_sentences/761_donation_unspecified')
dev_sets.append('kairos_schema_sentences/768_donation_unspecified')
dev_sets.append('kairos_schema_sentences/862_requestcommand_correspondence')

dev_sets.append('filtered_pengfei_scenario/scenario_en_124')
dev_sets.append('filtered_pengfei_scenario/scenario_en_62')
dev_sets.append('filtered_pengfei_scenario/scenario_en_70')

dev_sets.append('kairos_scenario_en/scenario_en_kairos_22')
dev_sets.append('kairos_scenario_en/scenario_en_kairos_35')
dev_sets.append('kairos_scenario_en/scenario_en_kairos_53')
dev_sets.append('kairos_scenario_en/scenario_en_kairos_64')
dev_sets.append('kairos_scenario_en/scenario_en_kairos_74')
dev_sets.append('kairos_scenario_en/scenario_en_kairos_95')

dev_sets.append('wiki_drone_strikes/wiki_drone_strikes_1_news_1')
dev_sets.append('wiki_drone_strikes/wiki_drone_strikes_1_news_2')
dev_sets.append('wiki_drone_strikes/wiki_drone_strikes_1_news_3')

dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_0')
dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_1')
dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_2')
dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_3')
dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_4')
dev_sets.append('wiki_ied_bombings_news_new/wiki_ied_bombings_3_news_5')

dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_0_article')
dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_1_news_0')
dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_1_news_1')
dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_1_news_8')
dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_1_news_9')
dev_sets.append('wiki_mass_car_bombings/wiki_mass_car_bombings_1_news_10')

dev_sets.append('quizlet_ta1/JC002YBVW')
dev_sets.append('quizlet_ta1/JC002YBVX')
dev_sets.append('quizlet_ta1/JC002YBVY')
dev_sets.append('quizlet_ta1/JC002YBVZ')
dev_sets.append('quizlet_ta1/JC002YBW0')
dev_sets.append('quizlet_ta1/JC002YBW2')

dev_sets.append('quizlet_ta2/JC002YBMJ')
dev_sets.append('quizlet_ta2/K0C03N4LR')
dev_sets.append('quizlet_ta2/K0C03N4MW')
dev_sets.append('quizlet_ta2/K0C03N4OM')

all_identified_events = dict()

for tmp_file_id in dev_sets:
    print('We are working on file:', tmp_file_id)
    document_path = 'data/all_kairos_data_10_04/' + tmp_file_id + '.rsd.txt'
    annotation_path = 'data/all_kairos_data_10_04/' + tmp_file_id + '.rsd.ann'
    annotation_result = load_annotation(annotation_path)
    if len(annotation_result['events']) == 0:
        print('We do not have annotation for this example')
        continue
    all_sentences = list()

    with open(document_path, 'r', encoding='utf-8') as f:

        for line in f:
            all_sentences.append(line)
    with open(document_path, 'r', encoding='utf-8') as f:
        full_document = f.read()
    previous_characters = 0
    identified_events = list()
    for tmp_s in tqdm(all_sentences):
        tmp_result = test_extractor.extract(tmp_s)
        if len(tmp_result) > 0:
            tmp_mapping = map_tokens_to_chars(tmp_result[0]['tokens'], tmp_s, previous_characters)
            new_identified_events = list()
            for old_e in tmp_result:
                new_e = old_e
                new_e['trigger']['position'] = (
                    tmp_mapping[old_e['trigger']['position'][0]][0],
                    tmp_mapping[old_e['trigger']['position'][1] - 1][1] + 1)
                new_arguments = list()
                for tmp_arg in old_e['arguments']:
                    new_tmp_arg = tmp_arg
                    new_tmp_arg['position'] = (
                        tmp_mapping[tmp_arg['position'][0]][0], tmp_mapping[tmp_arg['position'][1] - 1][1] + 1)
                    new_arguments.append(new_tmp_arg)
                new_e['arguments'] = new_arguments
                identified_events.append(new_e)
        previous_characters += len(tmp_s)
    all_identified_events[tmp_file_id.split('/')[1]] = identified_events

    num_trigger_gold += len(annotation_result['events'])
    for tmp_gold_e in annotation_result['events']:
        num_argument_gold += len(tmp_gold_e['arguments'])
        result_by_type[tmp_gold_e['event_type']]['num_trigger_gold'] += 1
        result_by_type[tmp_gold_e['event_type']]['num_argument_gold'] += len(tmp_gold_e['arguments'])
    num_trigger_predict += len(identified_events)
    for tmp_identified_e in identified_events:
        num_argument_predict += len(tmp_identified_e['arguments'])
        try:
            result_by_type[tmp_identified_e['trigger']['type']]['num_trigger_predict'] += 1
            result_by_type[tmp_identified_e['trigger']['type']]['num_argument_predict'] += len(
                tmp_identified_e['arguments'])
        except:
            print(tmp_identified_e['trigger'])
            print(tmp_identified_e)
            result_by_type[tmp_identified_e['trigger']['type']]['num_trigger_predict'] += 1
            result_by_type[tmp_identified_e['trigger']['type']]['num_argument_predict'] += len(
                tmp_identified_e['arguments'])
    for tmp_e in identified_events:
        for tmp_gold_e in annotation_result['events']:
            tmp_trigger_mention = annotation_result['mentions'][tmp_gold_e['trigger']]
            if tmp_e['trigger']['position'] == (tmp_trigger_mention['start_pos'], tmp_trigger_mention['end_pos']):
                num_trigger_identification_correct += 1
                result_by_type[tmp_gold_e['event_type']]['num_trigger_identification_correct'] += 1
                if tmp_e['trigger']['type'] == tmp_gold_e['event_type']:
                    num_trigger_classification_correct += 1
                    result_by_type[tmp_gold_e['event_type']]['num_trigger_classification_correct'] += 1
                for tmp_predicted_argument in tmp_e['arguments']:
                    for tmp_golden_argument in tmp_gold_e['arguments']:
                        tmp_argument_mention = annotation_result['mentions'][tmp_golden_argument['argument']]
                        # print(tmp_predicted_argument)
                        if tmp_predicted_argument['position'] == (
                                tmp_argument_mention['start_pos'], tmp_argument_mention['end_pos']):
                            num_argument_identification_correct += 1
                            result_by_type[tmp_gold_e['event_type']]['num_argument_identification_correct'] += 1
                            if tmp_predicted_argument['role'] == tmp_golden_argument['role']:
                                num_argument_classification_correct += 1
                                result_by_type[tmp_gold_e['event_type']]['num_argument_classification_correct'] += 1

    # print('gold')
    # print(annotation_result['events'])
    # for tmp_e in annotation_result['events']:
    #     tmp_trigger_mention = annotation_result['mentions'][tmp_e['trigger']]
    #     print(tmp_trigger_mention)
    #     print((tmp_trigger_mention['start_pos'], tmp_trigger_mention['end_pos']), full_document[tmp_trigger_mention['start_pos']:tmp_trigger_mention['end_pos']])
    # print('predicted')
    # print(identified_events)
    # for tmp_e in identified_events:
    #     print(tmp_e['trigger']['position'], full_document[tmp_e['trigger']['position'][0]: tmp_e['trigger']['position'][1]], tmp_e['trigger']['type'])
    # print('Number of identified events:', len(identified_events))
    # print(num_trigger_identification_correct)


p = num_trigger_identification_correct / num_trigger_predict
r = num_trigger_identification_correct / num_trigger_gold
f1 = (2 * p * r) / (p + r)
print('Trigger identification:', p, r, f1)

p = num_trigger_classification_correct / num_trigger_predict
r = num_trigger_classification_correct / num_trigger_gold
f1 = (2 * p * r) / (p + r)
print('Trigger classification:', p, r, f1)

print('num of gold', num_argument_gold)
print('num of correct', num_argument_identification_correct)
print('num of predict', num_argument_predict)
p = num_argument_identification_correct / num_argument_predict
r = num_argument_identification_correct / num_argument_gold
f1 = (2 * p * r) / (p + r)
print('Argument identification:', p, r, f1)

p = num_argument_classification_correct / num_argument_predict
r = num_argument_classification_correct / num_argument_gold
f1 = (2 * p * r) / (p + r)
print('Trigger classification:', p, r, f1)

gold_event_types_counting = dict()
identified_event_types_counting = dict()
print('Distribution of event types:')
print('event type', 'gold distribution', 'prediction distribution')
for tmp_event_type in event_types:
    gold_event_types_counting[tmp_event_type] = result_by_type[tmp_event_type]['num_trigger_gold']
    identified_event_types_counting[tmp_event_type] = result_by_type[tmp_event_type]['num_trigger_predict']
    print(tmp_event_type, result_by_type[tmp_event_type]['num_trigger_gold'],
          result_by_type[tmp_event_type]['num_trigger_predict'])

print_performance_by_type(result_by_type)

with open('tmp_identified_e_dev_set.json', 'w') as f:
    json.dump(all_identified_events, f)

print('end')
