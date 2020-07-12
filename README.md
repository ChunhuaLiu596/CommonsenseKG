# CommonsenseKG
Commonsense knowledge graph completion and alignment


# Data
Data set creation is in the folder of conceptnet-swow-data.
Run run_alignment.sh to generate the data.
```sh
$ cd conceptnet-swow-data/
$ ./run_alignment.sh
```

# Model 
Baseline model: TransE.    
Our model: TransER  (TransE + Relation Classifier)

# Train
Train the model:

```sh
$ cd cd OpenEA/run
$ ./run_rel.sh
```

The config is in OpenEA/run/args/iptranse_args_15K_rel.json.   
When predict_relation is False, run the baseline model TransE.  
When predict_relation is True, run the TransER model.
