# Week 3: Text-Model Feasibility Test

## Purpose

This exploratory test asks whether question text provides predictive
information beyond conventional structural features for a temporary outcome:
whether a Python question receives at least one answer within 24 hours. This is
not the project’s final measure of answer quality or successful help-seeking.

## Design

Questions from 2008--2014 were used for training, 2015 questions for
validation, and 2016 questions for final testing. The temporal split prevents a
model from being evaluated on data that precede its training data.

The structured baseline uses posting-time, question-level variables. TF-IDF
uses title, natural-language body, and tags as word and word-pair features.
MiniLM uses an embedding of the same text input, followed by logistic
regression. Question score and answer-derived variables are excluded because
they are not available at the time of posting.

## Preliminary results

On the held-out 2016 test set (n = 12,738), ROC-AUC was 0.629 for the
structured baseline, 0.728 for TF-IDF, and 0.755 for frozen MiniLM embeddings.
The results suggest that text contains more predictive information than the
available surface features alone. The MiniLM improvement over TF-IDF is modest,
so this does not establish that a language model is necessary or that model
performance is the project’s central contribution.

The associated metric table is `tables/model_comparison_metrics.csv`, and the
visualization is `figures/model_comparison_roc_auc.svg`.
