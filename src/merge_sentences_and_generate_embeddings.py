from util import *

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

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)
with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/verb_keywords.json', 'r') as f:
    framenet_verb_keywords = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/noun_keywords.json', 'r') as f:
    framenet_noun_keywords = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/other_keywords.json', 'r') as f:
    framenet_other_keywords = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/role_keywords.json', 'r') as f:
    framenet_role_keywords = json.load(f)

all_event_types = list(framenet_verb_keywords.keys())
all_role_types = list(framenet_role_keywords.keys())



# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertModel.from_pretrained('bert-large-uncased').to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)

model.eval()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-20)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/all_verb_reference_sentences.json', 'r', encoding='utf-8') as f:
    all_verb_keyword_to_sentences = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/all_noun_reference_sentences.json', 'r', encoding='utf-8') as f:
    all_noun_keyword_to_sentences = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/all_other_reference_sentences.json', 'r', encoding='utf-8') as f:
    all_other_keyword_to_sentences = json.load(f)

with open('/shared/hzhangal/Projects/zero-event-extraction/data_ACE2005/all_role_reference_sentences.json', 'r', encoding='utf-8') as f:
    all_role_keyword_to_sentences = json.load(f)

etype_to_distinct_embeddings = dict()
rtype_to_distinct_embeddings = dict()


selected_e_types = list()
selected_r_types = list()

for tmp_e_type in tqdm(all_event_types, desc='Loading predicate embeddings'):
    etype_to_distinct_embeddings[tmp_e_type] = list()
    for tmp_w in framenet_verb_keywords[tmp_e_type]:
        for tmp_example in all_verb_keyword_to_sentences[tmp_w][:10]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb.tolist()[0])
                # print(sent_emb.tolist()[0])
    for tmp_w in framenet_noun_keywords[tmp_e_type]:
        for tmp_example in all_noun_keyword_to_sentences[tmp_w][:10]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb.tolist()[0])
    for tmp_w in framenet_other_keywords[tmp_e_type]:
        for tmp_example in all_other_keyword_to_sentences[tmp_w][:10]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb.tolist()[0])
    if len(etype_to_distinct_embeddings[tmp_e_type]) > 0:
        selected_e_types.append(tmp_e_type)

for tmp_r_type in tqdm(all_role_types, desc='Loading argument embeddings'):
    rtype_to_distinct_embeddings[tmp_r_type] = list()
    for tmp_n in framenet_role_keywords[tmp_r_type]:
        selected_sentences = all_role_keyword_to_sentences[tmp_n][:10]
        for tmp_example in selected_sentences:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device, representation_type='mask')
            if not isinstance(sent_emb, str):
                rtype_to_distinct_embeddings[tmp_r_type].append(sent_emb.tolist()[0])
    if len(rtype_to_distinct_embeddings[tmp_r_type]) > 0:
        selected_r_types.append(tmp_r_type)

with open('data/all_reference_sentences.json', 'r', encoding='utf-8') as f:
    kairos_keywords_to_sentences = json.load(f)

for tmp_e_type in tqdm(trigger_keywords, desc='Loading predicate embeddings'):
    etype_to_distinct_embeddings[tmp_e_type] = list()
    for tmp_w in trigger_keywords[tmp_e_type]:
        for tmp_example in kairos_keywords_to_sentences[tmp_w][:10]:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device)
            if not isinstance(sent_emb, str):
                etype_to_distinct_embeddings[tmp_e_type].append(sent_emb.tolist()[0])

for tmp_r_type in tqdm(role_keywords, desc='Loading argument embeddings'):
    rtype_to_distinct_embeddings[tmp_r_type] = list()
    for tmp_n in role_keywords[tmp_r_type]:
        selected_sentences = kairos_keywords_to_sentences[tmp_n][:10]
        for tmp_example in selected_sentences:
            sent_emb = get_represetation(tmp_example[0], (tmp_example[1], tmp_example[2]),
                                         tokenizer, model, device, representation_type='mask')
            if not isinstance(sent_emb, str):
                rtype_to_distinct_embeddings[tmp_r_type].append(sent_emb.tolist()[0])
    # if len(rtype_to_distinct_embeddings[tmp_r_type]) > 0:
    #     selected_r_types.append(tmp_r_type)


with open('data/etype_to_distinct_embeddings.json', 'w') as f:
    json.dump(etype_to_distinct_embeddings, f)

with open('data/rtype_to_distinct_embeddings.json', 'w') as f:
    json.dump(rtype_to_distinct_embeddings, f)

with open('data/selected_e_types.json', 'w') as f:
    json.dump(selected_e_types, f)

with open('data/selected_r_types.json', 'w') as f:
    json.dump(selected_r_types, f)



