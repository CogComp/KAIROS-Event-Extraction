import json
import os
from util import *
import re

# import < your_code >


def Get_CogComp_SRL_results(input_sentence):


    # We then work on Celine's SRL system.
    # print('Extracting the events.')
    SRL_tokens = list()
    SRL_sentences = list()
    SRL_response = requests.get('http://dickens.seas.upenn.edu:4039/annotate', data=input_sentence)
    if SRL_response.status_code != 200:
        return None, None
    SRL_result = json.loads(SRL_response.text)
    SRL_tokens = SRL_result['tokens']
    SRL_sentences = SRL_result['sentences']
    return SRL_tokens, SRL_sentences
    # print('Match tokens.')



class EventExtraction(object):
    def annotate(self, text):
        input_paragraph = text
        headers = {'Content-type': 'application/json'}
        input_paragraph = re.sub(r'[\n]', ' ', input_paragraph)
        NER_response = requests.post('http://dickens.seas.upenn.edu:4022/ner/',
                                        json={"task": "ner", "text": input_paragraph}, headers=headers)
        if NER_response.status_code != 200:
            return {'error': 'The NER service is down.'}
        # SRL_response = requests.get('http://dickens.seas.upenn.edu:4039/annotate', data=input_paragraph)
        SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate',
                                        json={'sentence': input_paragraph})

        if SRL_response.status_code != 200:
            return {'error': 'The SRL service is down.'}
        SRL_tokens, SRL_sentences = Get_CogComp_SRL_results(input_paragraph)
        print(SRL_sentences['sentenceEndPositions'])
        sentences = list()
        sentences_by_char = list()
        for i, tmp_s_end_token in enumerate(SRL_sentences['sentenceEndPositions']):
            if i == 0:
                sentences.append(' '.join(SRL_tokens[:tmp_s_end_token]))
                sentences_by_char.append(SRL_tokens[:tmp_s_end_token])
            else:
                sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token]))
                sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token])
        if SRL_sentences['sentenceEndPositions'][-1] < len(SRL_tokens):
            sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:]))
            sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:])
        # sentences = input_paragraph.split('\n')
        print('Number of sentences:', len(sentences))
        previous_char = 0
        tmp_view_data = dict()
        tmp_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView'
        tmp_view_data['viewName'] = 'event_extraction'
        tmp_view_data['generator'] = 'cogcomp_kairos_event_ie_v1.0'
        tmp_view_data['score'] = 1.0
        tmp_view_data['constituents'] = list()
        tmp_view_data['relations'] = list()
        all_tokens = list()
        sentence_positions = list()
        previous_char = 0
        for s_id, tmp_s in enumerate(sentences):
            extracted_events = extractor.extract(tmp_s)
            print(extracted_events)
            if len(extracted_events) > 0:
                tmp_tokens = extracted_events[0]['tokens']
            else:
                tmp_tokens = tmp_s.split(' ')
            all_tokens += tmp_tokens
            sentence_positions.append(len(all_tokens))
            for tmp_event in extracted_events:
                trigger_start_token_id = tmp_event['trigger']['position'][0] + previous_char
                trigger_end_token_id = tmp_event['trigger']['position'][1] + previous_char

                trigger_consituent_position = len(tmp_view_data['constituents'])
                tmp_view_data['constituents'].append(
                    {'label': tmp_event['trigger']['type'], 'score': 1.0, 'start': trigger_start_token_id,
                        'end': trigger_end_token_id, 'properties': {
                        'SenseNumber': '01', 'sentence_id': s_id,
                                                                    'predicate': tmp_tokens[
                                                                                trigger_start_token_id:trigger_end_token_id]}})
                for tmp_argument in tmp_event['arguments']:
                    argument_start_token_id = tmp_argument['position'][0] + previous_char
                    argument_end_token_id = tmp_argument['position'][1] + previous_char
                    tmp_view_data['relations'].append(
                        {'relationName': tmp_argument['role'], 'srcConstituent': trigger_consituent_position,
                            'targetConstituent': len(tmp_view_data['constituents'])})
                    tmp_view_data['constituents'].append(
                        {'label': tmp_argument['role'], 'score': 1.0, 'start': argument_start_token_id,
                            'end': argument_end_token_id, 'entity_type': tmp_argument['entity_type']})
            previous_char += len(sentences_by_char[s_id])

            event_ie_view = dict()
            event_ie_view['viewName'] = 'Event_extraction'
            event_ie_view['viewData'] = [tmp_view_data]

            token_view = dict()
            token_view['viewName'] = 'TOKENS'
            tmp_token_view_data = dict()
            tmp_token_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView'
            tmp_token_view_data['viewName'] = 'TOKENS'
            tmp_token_view_data['generator'] = 'Cogcomp-SRL'
            tmp_token_view_data['score'] = 1.0
            tmp_token_view_data['constituents'] = list()
            for i, tmp_token in enumerate(all_tokens):
                tmp_token_view_data['constituents'].append({'label': tmp_token, 'score': 1.0, 'start': i, 'end': i+1})
            token_view['viewData'] = tmp_token_view_data

            result = dict()
            result['corpusId'] = ''
            result['id'] = ''
            result['text'] = text
            result['tokens'] = SRL_tokens
            result['sentences'] = SRL_sentences
            result['views'] = [token_view, event_ie_view]



    # return resulting JSON
        return result

    def annotateMain(self, mode = "content", content="", filename="", input_directory="", output_directory=""):
        if mode == "content":
            output = self.annotate(text=content)
            print(output)
            return output
        elif mode == "file":
            content = open(filename, "r").read()
            output = self.annotate(content)
            print(output)
            return output
        
        elif mode == "directory":
            file_list = os.listdir(input_directory)
            for filename in file_list:
                content = open(input_directory + "/" + filename, "r").read()
                annJson =  self.annotate(content)
                print("_" * 50)
                print(filename)
                print("_" * 50)
                print(annJson)
                print("_" * 50)


    # def annotateMain(self, args):
    #     if args.mode == "content":
    #         output = self.annotate(sentence=args.content)
    #         print(output)
    #         return output
    #     elif args.mode == "file":
    #         content = open(args.filename, "r").read()
    #         output = self.annotate(content)
    #         print(output)
    #         return output
        
    #     elif args.mode == "directory":
    #         file_list = os.listdir(args.input_directory)
    #         for filename in file_list:
    #             content = open(args.input_directory + "/" + filename, "r").read()
    #             annJson =  self.annotate(content)
    #             print("_" * 50)
    #             print(filename)
    #             print("_" * 50)
    #             print(annJson)
    #             print("_" * 50)



if __name__ == '__main__':
    print("")
    # INITIALIZE YOUR MODEL HERE
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device:', device)

    extractor = CogcompKairosEventExtractorTest(device, 'mbert')

    text = "A firefighter and his crew battled to keep the raging Glass Fire from devastating an upmarket Napa Valley vineyard. The firefiighter denies lighting backfires which consume fuel in a wildfire's path but admits his team failed to advise Cal Fire, the state's fire agency that it was in the evacuated area, as required by law."
    evObj = EventExtraction()
    # evObj.annotateMain(mode = "content", content=text, filename="", input_directory="", output_directory="")
    evObj.annotateMain(mode = "directory", content="", filename="", input_directory="input/", output_directory="")
    # evObj.annotateMain(args)
