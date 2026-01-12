Originaly forked from: [czsl](https://github.com/ExplainableML/czsl), but most of the code is not used in this project.

# ALIGN fine-tuning

This project tries to fine-tune a [ALIGN](https://huggingface.co/docs/transformers/model_doc/align) for the purpes for
word spotting. The goal is to get a generlized model that can do some word spotting on unseen languages.

The original idea were to incorperate this as the image endocer used in [czsl](https://github.com/ExplainableML/czsl) to
get it to do word spotting but with better preformence and even better generlization to diffrent languages.

# Experiments

## fold_3_aug_sqare

Align

| LR    | Eps  | Batch size | Accum | Head size | Optimizer | Lr scheduler | lr_exploration_factor | Loss function | Description | Folder number | Status |
|-------|------|------------|-------|-----------|-----------|--------------|-----------------------|---------------|-------------|---------------|--------|
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 5             | X      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 6             | X      |
| 1e-4  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 7             | X      |
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 11            | X      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 12            | X      |
| 1e-4  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 13            | X      |
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 18            | X      |
| 1e-6  | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Short desc  | 23            | F      |
| 1e-6  | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 24            | F      |
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 22            | F      |
| 1e-10 | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 31            | R      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | lr_scheduler | 1.5                   | Supcon        | Long desc   | 28            | R      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | lr_scheduler | 2.0                   | Supcon        | Long desc   | 29            | R      |

CLIP

