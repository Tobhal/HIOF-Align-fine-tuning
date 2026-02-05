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
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 1             | D      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 2             | D      |
| 1e-4  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 3             | D      |
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 4             | D      |
| 1e-5  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 5             | D      |
| 1e-4  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 6             | D      |
| 1e-6  | 1e-8 | 32         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Indices     | 7             | D      |
| 1e-6  | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Short desc  | 8             | F      |
| 1e-6  | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 9             | F      |
| 1e-6  | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 10            | F      |
| 1e-10 | 1e-8 | 32         | 8     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Long desc   | 11            | F      |
| 1e-5  | 1e-8 | 32         | 8     | 1024      | Adam      | lr_scheduler | 1.5                   | contrastive   | Long desc   | 12            | D      |
| 1e-5  | 1e-8 | 32         | 8     | 1024      | Adam      | lr_scheduler | 2.0                   | contrastive   | Long desc   | 13            | D      |
| 1e-10 | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 31            | R      |

CLIP (Maximize)

| LR    | Eps  | Batch size | Accum | Head size | Optimizer | Lr scheduler | lr_exploration_factor | Loss function | Description | Folder number | Status |
|-------|------|------------|-------|-----------|-----------|--------------|-----------------------|---------------|-------------|---------------|--------|
| 1e-6  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 2             | R      |
| 1e-5  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 3             | R      |
| 1e-4  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 4             | R      |
| 1e-10 | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 8             | R      |

CLIP (Minimize)

| LR    | Eps  | Batch size | Accum | Head size | Optimizer | Lr scheduler | lr_exploration_factor | Loss function | Description | Folder number | Status |
|-------|------|------------|-------|-----------|-----------|--------------|-----------------------|---------------|-------------|---------------|--------|
| 1e-6  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 5             | R      |
| 1e-5  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 6             | R      |
| 1e-4  | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 7             | R      |
| 1e-10 | 1e-8 | 16         | 4     | 1024      | Adam      | Cosine       | 1.5                   | Supcon        | Word        | 9             | R      |
