# TODO

Open tasks. See also [Known issues](README.md#known-issues).

## 1. Modernise

- [x] Replace Poetry with uv and the loose scripts with the package `src/sdg_classifier` and the commands `prepare`,
      `train` and `predict`
- [x] Update to current dependencies (PyTorch 2.14, Transformers 5, datasets 5, PEFT 0.21) and Python 3.12+
- [x] Remove unused dependencies (sentence-transformers, KeyBERT, BERTopic, NLTK, UMAP, pydantic and others)
- [x] Apply the class weights in the loss; the 2024 scripts set them on attributes the models never read
- [x] Replace scikit-multilearn (unmaintained) with iterative-stratification and seed the split
- [x] Merge `duplication.py` into `prepare --one-label-per-row` and `train_lora.py` into `train --lora`
- [x] LoRA: train the classification head too, use a learning rate for adapters (2e-4 instead of 2e-6), and save the
      merged model
- [x] Remove the `push_to_hub` call at the end of training
- [x] Test on the real data: `prepare`, a short run of `train` and `train --lora`, and `predict` with the 2024 model
- [ ] Retrain the full model with class weights and update [Results](README.md#results)
- [ ] Optionally publish the trained model on the Hugging Face Hub

## 2. Publish

- [x] Add the GPLv3 license text named in the README
- [x] Credit the Zenodo dataset and OpenAlex
- [x] Check the files and the git history for secrets (none found)
- [x] Replace the old university e-mail address in the history
- [x] Rename the GitHub repository from `sdg_classifier` to `sdg-classifier`
