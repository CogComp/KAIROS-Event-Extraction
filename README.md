# Kairos Event Extraction

## Description
The zero-shot event extraction demo is used to show the effectiveness of zero-shot label-aware event trigger and argument classification model. We propose a zero-shot event extraction approach, which first identifies events with existing tools (e.g., SRL) and then maps them to a given taxonomy of event types in a zero-shot manner. Specifically, we leverage label representations induced by pre-trained language models, and map identified events to the target types via representation similarity. To semantically type the events’ arguments, we further use the definition of the events (e.g., argument of type 'Victim' appears as the argument of event of type 'Attack') as global constraints to regularize the prediction. The candidate event ontology we use for this demo is from ACE-2005.

## Environment Setup
1. Setup the environment with the environment.yml file (`conda env create -f environment.yml`)
2. Downgrade the xlrd package to 1.2.0 with `pip uninstall xlrd` and then `pip install xlrd==1.2.0`

## Downloads

Download the following two files from server with following wget commands and put them under the `data/` folder.
- `wget https://www.seas.upenn.edu/~hzhangal/etype_to_distinct_embeddings.json` 
- `wget https://www.seas.upenn.edu/~hzhangal/rtype_to_distinct_embeddings.json` 

## Run
To run service enter `python backend.py`

## Sample Input/Output
1. Input example: curl -d '{"sentence": "The president of the USA holds a lot of power."}' -H "Content-Type: application/json" -X POST http://localhost:20203/annotate
2. Output example: {"corpusId": "", "id": "", "text": "The president of the USA holds a lot of power.", "tokens": ["The", "president", "of", "the", "USA", "holds", "a", "lot", "of", "power."], "sentences": {"generator": "UnsupervisedEventExtraction", "score": 1.0, "sentenceEndPositions": [46]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "The", "score": 1.0, "start": 0, "end": 1}, {"label": "president", "score": 1.0, "start": 1, "end": 2}, {"label": "of", "score": 1.0, "start": 2, "end": 3}, {"label": "the", "score": 1.0, "start": 3, "end": 4}, {"label": "USA", "score": 1.0, "start": 4, "end": 5}, {"label": "holds", "score": 1.0, "start": 5, "end": 6}, {"label": "a", "score": 1.0, "start": 6, "end": 7}, {"label": "lot", "score": 1.0, "start": 7, "end": 8}, {"label": "of", "score": 1.0, "start": 8, "end": 9}, {"label": "power.", "score": 1.0, "start": 9, "end": 10}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [], "relations": []}]}]}

## Demo


<img src=/img/Event-Extraction-Demo.png>
