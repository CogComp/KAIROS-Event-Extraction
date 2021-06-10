import cherrypy
import cherrypy_cors
import json
import os
from util import *


# import < your_code >
def str_to_tuple(string):
    a0 = int(string.split(", ")[0].replace("(", ""))
    a1 = int(string.split(", ")[1].replace(")", ""))
    return (a0, a1)


class MyWebService(object):

    @cherrypy.expose
    def index(self):
        return open('html/index.html', encoding='utf-8')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def annotate(self):

        hasJSON = True
        result = {"status": "false"}
        try:
            # get input JSON
            data = cherrypy.request.json
        except:
            hasJSON = False
            result = {"error": "invalid input"}

        if hasJSON:
            # process input

            input_sentence = data['sentence']
            tokens = data['tokens']
            identified_trigger_positions = data['identified_trigger_positions']

            detected_mentions_str = data['detected_mentions']
            trigger_to_arguments_str = data['trigger_to_arguments']
            detected_mentions = {}
            trigger_to_arguments = {}
            for item in detected_mentions_str:
                detected_mentions[str_to_tuple(item)] = detected_mentions_str[item]
            for item in trigger_to_arguments_str:
                trigger_to_arguments[str_to_tuple(item)] = trigger_to_arguments_str[item]

            tmp_view_data = dict()
            tmp_view_data[
                'viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView'
            tmp_view_data['viewName'] = 'event_extraction'
            tmp_view_data['generator'] = 'cogcomp_kairos_event_ie_v1.0'
            tmp_view_data['score'] = 1.0
            tmp_view_data['constituents'] = list()
            tmp_view_data['relations'] = list()
            all_tokens = list()
            sentence_positions = [len(input_sentence)]

            extracted_events = extractor.extract_with_annotation(input_sentence, tokens, detected_mentions,
                                                                 identified_trigger_positions, trigger_to_arguments)

            if len(extracted_events) > 0:
                tmp_tokens = extracted_events[0]['tokens']
            else:
                tmp_tokens = input_sentence.split(' ')
                all_tokens += tmp_tokens
            for tmp_event in extracted_events:
                trigger_start_token_id = tmp_event['trigger']['position'][0]
                trigger_end_token_id = tmp_event['trigger']['position'][1]

                trigger_consituent_position = len(tmp_view_data['constituents'])
                tmp_view_data['constituents'].append(
                    {'label': tmp_event['trigger']['type'], 'score': 1.0, 'start': trigger_start_token_id,
                     'end': trigger_end_token_id, 'properties': {
                        'SenseNumber': '01', 'sentence_id': 0,
                        'predicate': tmp_tokens[
                                     trigger_start_token_id:trigger_end_token_id]}})
                for tmp_argument in tmp_event['arguments']:
                    argument_start_token_id = tmp_argument['position'][0]
                    argument_end_token_id = tmp_argument['position'][1]
                    tmp_view_data['relations'].append(
                        {'relationName': tmp_argument['role'], 'srcConstituent': trigger_consituent_position,
                         'targetConstituent': len(tmp_view_data['constituents'])})
                    tmp_view_data['constituents'].append(
                        {'label': tmp_argument['role'], 'score': 1.0, 'start': argument_start_token_id,
                         'end': argument_end_token_id, 'entity_type': tmp_argument['entity_type']})

            event_ie_view = dict()
            event_ie_view['viewName'] = 'Event_extraction'
            event_ie_view['viewData'] = [tmp_view_data]

            token_view = dict()
            token_view['viewName'] = 'TOKENS'
            tmp_token_view_data = dict()
            tmp_token_view_data[
                'viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView'
            tmp_token_view_data['viewName'] = 'TOKENS'
            tmp_token_view_data['generator'] = 'Cogcomp-SRL'
            tmp_token_view_data['score'] = 1.0
            tmp_token_view_data['constituents'] = list()
            for i, tmp_token in enumerate(all_tokens):
                tmp_token_view_data['constituents'].append({'label': tmp_token, 'score': 1.0, 'start': i, 'end': i + 1})
            token_view['viewData'] = tmp_token_view_data

            result = dict()
            result['corpusId'] = ''
            result['id'] = ''
            result['text'] = data['text']
            result['tokens'] = all_tokens
            result['sentences'] = {'generator': 'UnsupervisedEventExtraction', 'score': 1.0,
                                   'sentenceEndPositions': sentence_positions}
            result['views'] = [token_view, event_ie_view]

        # return resulting JSON
        return result


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
    # IN ORDER TO KEEP IT IN MEMORY
    print("Starting rest service...")

    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': 'public'
        },
        '/css': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': "public/css"
        },
        '/js': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': "public/js"
        },
    }
    print(f"Starting rest service...{conf}")
    config = {'server.socket_host': '0.0.0.0'}
    cherrypy.config.update(config)
    cherrypy.config.update({'server.socket_port': 20203})
    cherrypy.quickstart(MyWebService(), '/', conf)