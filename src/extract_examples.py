from util import *

sp = spacy.load('en_core_web_sm')

all_keywords = list()
for tmp_trigger in trigger_keywords:
    all_keywords += trigger_keywords[tmp_trigger]
for tmp_role in role_keywords:
    all_keywords += role_keywords[tmp_role]
all_keywords = list(set(all_keywords))

print(trigger_keywords)
print(role_keywords)

def collect_examples(folder_id):
    tmp_sp = spacy.load('en_core_web_sm')
    role_keywords_to_sentences = dict()
    for tmp_keyword in all_keywords:
        role_keywords_to_sentences[tmp_keyword] = list()
    tmp_folder_path = '/shared/corpora-tmp/annotated_nyt/' + str(folder_id + 1)
    if not os.path.isdir(tmp_folder_path):
        return None
    tmp_file_names = os.listdir(tmp_folder_path)
    for tmp_f_name in tqdm(tmp_file_names, desc='Folder ' + str(folder_id + 1)):
        tmp_full_f_name = tmp_folder_path + '/' + tmp_f_name
        with open(tmp_full_f_name, 'r', encoding='utf-8') as f:
            current_document = json.load(f)
        sentences = current_document['text'].split('\n')
        for tmp_s in sentences:
            parsed_sentence = tmp_sp(tmp_s)
            all_tokens = list()
            for tmp_token in parsed_sentence:
                all_tokens.append(tmp_token.text)
            for tmp_token in parsed_sentence:
                if 'NN' == tmp_token.tag_[:2]:
                    if tmp_token.lemma_ in all_keywords:
                        role_keywords_to_sentences[tmp_token.lemma_].append((all_tokens, tmp_token.i, tmp_token.i + 1))
    with open('tmp/' + str(folder_id + 1) + '.json', 'w', encoding='utf-8') as tmp_f:
        json.dump(role_keywords_to_sentences, tmp_f)


all_ids = range(899)
# all_ids = range(20)
workers = Pool(20)
for tmp_id in all_ids:
    workers.apply_async(collect_examples, args=(tmp_id,))
workers.close()
workers.join()

all_keyword_to_sentences = dict()
for tmp_key_word in all_keywords:
    all_keyword_to_sentences[tmp_key_word] = list()

for tmp_id in tqdm(all_ids, desc='Merging'):
    try:
        with open('tmp/' + str(tmp_id + 1) + '.json', 'r', encoding='utf-8') as tmp_f:
            tmp_keyword_to_sentences = json.load(tmp_f)

        for tmp_role_key_word in tmp_keyword_to_sentences:
            all_keyword_to_sentences[tmp_role_key_word] += tmp_keyword_to_sentences[tmp_role_key_word]
    except:
        print('Something is wrong with folder:', tmp_id + 1)

print('role keywords')
for tmp_role_key_word in all_keyword_to_sentences:
    print(tmp_role_key_word, ':', len(all_keyword_to_sentences[tmp_role_key_word]))

with open('data/all_reference_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(all_keyword_to_sentences, f)

shutil.rmtree('tmp')
os.mkdir('tmp')
