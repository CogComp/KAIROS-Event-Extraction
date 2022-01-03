from util import *
from document_reader import *
import os

folder_name = '/shared/kairos/Data/LDC2020E30_KAIROS_Quizlet_3_Source_Data_and_Graph_G/data/source/ltf/ltf/'
documents = list()
for tmp_file_name in os.listdir(folder_name):
    if 'xml' in tmp_file_name:
        extracted_data = ltf_reader(folder_name, tmp_file_name)
        documents.append(extracted_data)

    # sentences = list()
    # for tmp_s in extracted_data['sentences']:
    #     sentences.append(tmp_s['content'])

    # new_file_name = tmp_file_name.replace('.xml', '.txt')
    # with open('data/quizlet3/' + new_file_name, 'w', encoding='utf-8') as f:
    #     for s in sentences:
    #         f.write(s)
    #         f.write('\n')

# print('done')


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

results = list()
for tmp_document in documents:
    print('We are working on document:', tmp_document["doc_id"])
    extracted_results = list()
    for tmp_s in tqdm(tmp_document['sentences']):
        extracted_results.append(test_extractor.extract(tmp_s['content']))
    tmp_document['event_extraction_results'] = extracted_results
    results.append(tmp_document)

with open('data/quizlet3/data_after_event_extraction.json', 'w') as f:
    json.dump(results, f)





print('end')
