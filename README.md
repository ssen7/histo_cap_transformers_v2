# histo_cap_transformers_v2

## Updates

Streamlined code for histopath image captioning. No need to download the full HIPT code. Just refer to [full_inference_pipeline](./full_inference_pipeline/full_inference_pipeline.ipynb) for a demonstration on how the captioning pipeline works.

WARNING! Experimental code
## Log
### Date: Jan 24, 2024
Added a full inference pipeline with added visualization code.

### Date: Jan 22, 2024
Experimental code for new approach to generate histopathology captions.

This is experimental code for using HIPT for code generation. Here I am using the generated (n_patchx256x384) and (n_patchx1x192) encodings to tokenize the WSI with n_patch tokens and pass it to the BERTLMHead decoder. So far it looks promising.

Future features:
1. Add contrastive loss.
2. Add a thumbnail encoder.