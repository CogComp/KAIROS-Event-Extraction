# Kairos Event Extraction

## Summary
Identifying events and mapping them to a pre-defined taxonomy of event types has long been an important NLP problem. Most previous work has relied heavily on labor-intensive, domain-specific, annotation, ignoring the semantic meaning of the event types labels. Consequently, the learned models cannot effectively generalize to new label taxonomies and domains. We propose a zero-shot event extraction approach, which first identifies events with existing tools (e.g., SRL) and then maps them to a given taxonomy of event types in a zero-shot manner. Specifically, we leverage label representations induced by pre-trained language models, and map identified events to the target types via representation similarity. To semantically type the events’ arguments, we further use the definition of the events (e.g., argument of type “Victim” appears as the argument of event of type “Attack”) as global constraints to regularize the prediction. The proposed approach is shown to be very effective on the ACE-2005 dataset, which has 33 trigger and 22 argument types. Without using any annotation, we successfully map 83% of the triggers and 54% of the arguments to the semantic correct types, almost doubling the performance of previous zero-shot approaches.

## Environment Setup
1. Setup the environment with the environment.yml file (`conda env create -f environment.yml`)
2. Downgrade the `xlrd` package to 1.2.0 with `pip uninstall xlrd` and then `pip install xlrd==1.2.0`

## Downloads
Download the following two files from server with wget commands and save them in the `data/` folder.
- `wget https://www.seas.upenn.edu/~hzhangal/etype_to_distinct_embeddings.json` 
- `wget https://www.seas.upenn.edu/~hzhangal/rtype_to_distinct_embeddings.json` 

## Run
To run service enter `python backend.py`

## Sample Input/Output
<b>Input Sample:</b> <br/>
curl -d '{"sentence": "The World Health Organization on Wednesday declared the novel coronavirus outbreak a pandemic."}' -H "Content-Type: application/json" -X POST http://localhost:20203/annotate <br /><br />
<b>Sample Output:</b><br/>
{'corpusId': '', 'id': '', 'text': 'The World Health Organization on Wednesday declared the novel coronavirus outbreak a pandemic. ', 'tokens': ['The', 'World', 'Health', 'Organization', 'on', 'Wednesday', 'declared', 'the', 'novel', 'coronavirus', 'outbreak', 'a', 'pandemic', '.'], 'sentences': {'generator': 'srl_pipeline', 'score': 1.0, 'sentenceEndPositions': [14]}, 'views': [{'viewName': 'TOKENS', 'viewData': {'viewType': 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView', 'viewName': 'TOKENS', 'generator': 'Cogcomp-SRL', 'score': 1.0, 'constituents': [{'label': 'The', 'score': 1.0, 'start': 0, 'end': 1}, {'label': 'World', 'score': 1.0, 'start': 1, 'end': 2}, {'label': 'Health', 'score': 1.0, 'start': 2, 'end': 3}, {'label': 'Organization', 'score': 1.0, 'start': 3, 'end': 4}, {'label': 'on', 'score': 1.0, 'start': 4, 'end': 5}, {'label': 'Wednesday', 'score': 1.0, 'start': 5, 'end': 6}, {'label': 'declared', 'score': 1.0, 'start': 6, 'end': 7}, {'label': 'the', 'score': 1.0, 'start': 7, 'end': 8}, {'label': 'novel', 'score': 1.0, 'start': 8, 'end': 9}, {'label': 'coronavirus', 'score': 1.0, 'start': 9, 'end': 10}, {'label': 'outbreak', 'score': 1.0, 'start': 10, 'end': 11}, {'label': 'a', 'score': 1.0, 'start': 11, 'end': 12}, {'label': 'pandemic', 'score': 1.0, 'start': 12, 'end': 13}, {'label': '.', 'score': 1.0, 'start': 13, 'end': 14}]}}, {'viewName': 'Event_extraction', 'viewData': [{'viewType': 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView', 'viewName': 'event_extraction', 'generator': 'cogcomp_kairos_event_ie_v1.0', 'score': 1.0, 'constituents': [{'label': 'Disaster:DiseaseOutbreak:Unspecified', 'score': 1.0, 'start': 9, 'end': 10, 'properties': {'SenseNumber': '01', 'sentence_id': 0, 'predicate': ['coronavirus']}}, {'label': 'Disease', 'score': 1.0, 'start': 9, 'end': 10, 'entity_type': 'mhi'}, {'label': 'Disaster:DiseaseOutbreak:Unspecified', 'score': 1.0, 'start': 10, 'end': 11, 'properties': {'SenseNumber': '01', 'sentence_id': 0, 'predicate': ['outbreak']}}, {'label': 'Disease', 'score': 1.0, 'start': 9, 'end': 10, 'entity_type': 'mhi'}, {'label': 'Disaster:DiseaseOutbreak:Unspecified', 'score': 1.0, 'start': 12, 'end': 13, 'properties': {'SenseNumber': '01', 'sentence_id': 0, 'predicate': ['pandemic']}}, {'label': 'Disease', 'score': 1.0, 'start': 9, 'end': 10, 'entity_type': 'mhi'}], 'relations': [{'relationName': 'Disease', 'srcConstituent': 0, 'targetConstituent': 1}, {'relationName': 'Disease', 'srcConstituent': 2, 'targetConstituent': 3}, {'relationName': 'Disease', 'srcConstituent': 4, 'targetConstituent': 5}]}]}]}

<br/>

## Demo Visualization
![image](https://user-images.githubusercontent.com/22654200/151893966-7ac5f3b0-1af3-4b1f-aad1-e2abd851ed02.png)


## Link to Web Demo
https://cogcomp.seas.upenn.edu/page/demo_view/EventExtraction


