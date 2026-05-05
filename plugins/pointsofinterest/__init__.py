"""Autism plugin for Hoploy.

This plugin registers domain-specific components for the autism-friendly
recommendation use-case.

Modules
-------
sensory
    Sensory-feature compatibility logic (aversions, Likert scales,
    ``user_feature_mask``, ``user_sample_compatible_features``).
input_preparation
    Builds raw input sequences and configures logit processors for
    *existing-user* and *zero-shot* modes (``prepare_recommender_and_
    raw_inputs_existing_user``, ``prepare_recommender_and_raw_inputs_
    zero_shot``, ``reset_logits_processors``).
unpacker
    Converts post-processed sequences into human-readable
    recommendations with Italian explanations
    (``unpack_recommendation_sequences_tuples``).
"""
