# Data

- swisstext-2024-sharedtask = use `sh data/swisstext-2024-sharedtask/download-data.sh`
- osdg.csv = downloaded from Tobias' repo, should contain OSDG community dataset
- zo_up.csv = upsampled Zora data with Osdg data
- zo_up_sdg0.jsonl = comes from march 2024 swisstext training data and a selection of
  swisstext test data (in order to match 56 items for all classes)
- zo_up_sdg17.jsonl = comes from synthetic data from swisstext contributors from chur
  (openai and llama2) plus freshly generated synthetic data by GPT 4o (when using the
  first synthetic data as samples, in order to match 56 items for all classes)
