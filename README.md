# Kairos Event Extraction

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
curl -d '{"sentence": "The president of the USA holds a lot of power."}' -H "Content-Type: application/json" -X POST http://localhost:20203/annotate <br /><br />
<b>Sample Output:</b><br/>
{"corpusId": "", "id": "", "text": "The president of the USA holds a lot of power.", "tokens": ["The", "president", "of", "the", "USA", "holds", "a", "lot", "of", "power."], "sentences": {"generator": "UnsupervisedEventExtraction", "score": 1.0, "sentenceEndPositions": [46]}, "views": [{"viewName": "TOKENS", "viewData": {"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView", "viewName": "TOKENS", "generator": "Cogcomp-SRL", "score": 1.0, "constituents": [{"label": "The", "score": 1.0, "start": 0, "end": 1}, {"label": "president", "score": 1.0, "start": 1, "end": 2}, {"label": "of", "score": 1.0, "start": 2, "end": 3}, {"label": "the", "score": 1.0, "start": 3, "end": 4}, {"label": "USA", "score": 1.0, "start": 4, "end": 5}, {"label": "holds", "score": 1.0, "start": 5, "end": 6}, {"label": "a", "score": 1.0, "start": 6, "end": 7}, {"label": "lot", "score": 1.0, "start": 7, "end": 8}, {"label": "of", "score": 1.0, "start": 8, "end": 9}, {"label": "power.", "score": 1.0, "start": 9, "end": 10}]}}, {"viewName": "Event_extraction", "viewData": [{"viewType": "edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView", "viewName": "event_extraction", "generator": "cogcomp_kairos_event_ie_v1.0", "score": 1.0, "constituents": [], "relations": []}]}]}
