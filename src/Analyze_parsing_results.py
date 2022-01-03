import ujson as json


with open('data/quizlet3/data_after_event_extraction.json', 'r') as f:
    tmp_results = json.load(f)


event_counter = 0
for tmp_document in tmp_results:
    for tmp_prediction_by_sentence in tmp_document['event_extraction_results']:
        event_counter += len(tmp_prediction_by_sentence)

print('Number of extracted events:', event_counter)
print('Number of sentences:', len(tmp_results))
