# from allennlp.predictors.predictor import Predictor
import ujson as json
import spacy
import pandas
import requests

# print('Start to load the SRL model.')
# SRL_predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
# NER_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
# print('Finish loading the SRL model.')
#
#
# def Verbal_SRL_parsing(input_sentence):
#     tmp_result = SRL_predictor.predict_batch_json([{'sentence': input_sentence}])[0]
#     return tmp_result
#
# def NER_parsing(input_sentence):
#     try:
#         tmp_result = NER_predictor.predict(sentence=input_sentence)
#         return tmp_result
#     except:
#         return {'logits': [], 'mask': [], 'tags': [], 'words': []}
#
#
# SRL_result = Verbal_SRL_parsing('I love Beijing')
# NER_result = NER_parsing('I love Beijing')
#
# print(SRL_result)
# print(NER_result)

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
                role_to_type[row[tmp_key]] = row[tmp_constraint_key].split(',')
                role_keywords[row[tmp_key]] = row[tmp_keywords].split('$$')
        event_schema_to_roles[event_name] = roles
    role_types = list(set(role_types))

    return {'event_types': event_types, 'role_types': role_types, 'event_schema_to_roles': event_schema_to_roles,
            'role_to_type': role_to_type, 'trigger_keywords': trigger_keywords, 'role_keywords': role_keywords}



### Testing Celine's SRL API


# SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', data='Bob attacks Kevin. I love you.')

SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', json = {'sentence': 'Bob attacks Kevin. \n I love you.'})

print(SRL_response.status_code)
SRL_test_result = json.loads(SRL_response.text)
print(SRL_test_result)
#
# headers = {'Content-type':'application/json'}
# NER_response = requests.post('http://dickens.seas.upenn.edu:8099/ner/', json={"task":"ner", "text" : "beijing is a city"}, headers=headers)
# print(NER_response.status_code)
# NER_test_result = json.loads(NER_response.text)
# print(NER_test_result)
# print('end')

