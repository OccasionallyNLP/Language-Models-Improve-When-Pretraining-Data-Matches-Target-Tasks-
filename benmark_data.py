from datasets import load_dataset
import random
import json
from tqdm import tqdm
length_of_dataset = 1500
total_data = []
# mmlu
print('mmlu')
ds = load_dataset("cais/mmlu", "all", split='test')
subset = ds.shuffle(seed=42).select(range(length_of_dataset))
total_data.extend(subset['question'])

# mmlu pro
print('mmlu pro')
ds = load_dataset("TIGER-Lab/MMLU-Pro", split='test')
subset = ds.shuffle(seed=42).select(range(length_of_dataset))
total_data.extend(subset['question'])

# winogrande
print('winogrande')
ds = load_dataset("allenai/winogrande", 'winogrande_xl', split='test')
subset = ds.shuffle(seed=42).select(range(length_of_dataset))
total_data.extend(subset['sentence'])

# ai2 arc
print('ai2 arc')
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split='test')
subset = ds.shuffle(seed=42).select(range(length_of_dataset))
total_data.extend(subset['question'])

# hellaswag
print('hellaswag')
ds = load_dataset("Rowan/hellaswag", split='test')
subset = ds.shuffle(seed=42).select(range(length_of_dataset))
total_data.extend(subset['ctx'])

# default
print('bigbench')
domain_names = ['abstract_narrative_understanding_zero_shot', 'anachronisms_zero_shot', 'analogical_similarity_zero_shot', 'analytic_entailment_zero_shot', 'arithmetic_zero_shot', 'ascii_word_recognition_zero_shot', 'authorship_verification_zero_shot', 'auto_categorization_zero_shot', 'auto_debugging_zero_shot', 'bbq_lite_json_zero_shot', 'bridging_anaphora_resolution_barqa_zero_shot', 'causal_judgment_zero_shot', 'cause_and_effect_zero_shot', 'checkmate_in_one_zero_shot', 'chess_state_tracking_zero_shot', 'chinese_remainder_theorem_zero_shot', 'cifar10_classification_zero_shot', 'code_line_description_zero_shot', 'codenames_zero_shot', 'color_zero_shot', 'common_morpheme_zero_shot', 'conceptual_combinations_zero_shot', 'conlang_translation_zero_shot', 'contextual_parametric_knowledge_conflicts_zero_shot', 'crash_blossom_zero_shot', 'crass_ai_zero_shot', 'cryobiology_spanish_zero_shot', 'cryptonite_zero_shot', 'cs_algorithms_zero_shot', 'dark_humor_detection_zero_shot', 'date_understanding_zero_shot', 'disambiguation_qa_zero_shot', 'discourse_marker_prediction_zero_shot', 'disfl_qa_zero_shot', 'dyck_languages_zero_shot', 'elementary_math_qa_zero_shot', 'emoji_movie_zero_shot', 'emojis_emotion_prediction_zero_shot', 'empirical_judgments_zero_shot', 'english_proverbs_zero_shot', 'english_russian_proverbs_zero_shot', 'entailed_polarity_hindi_zero_shot', 'entailed_polarity_zero_shot', 'epistemic_reasoning_zero_shot', 'evaluating_information_essentiality_zero_shot', 'fact_checker_zero_shot', 'fantasy_reasoning_zero_shot', 'few_shot_nlg_zero_shot', 'figure_of_speech_detection_zero_shot', 'formal_fallacies_syllogisms_negation_zero_shot', 'gem_zero_shot', 'gender_inclusive_sentences_german_zero_shot', 'general_knowledge_zero_shot', 'geometric_shapes_zero_shot', 'goal_step_wikihow_zero_shot', 'gre_reading_comprehension_zero_shot', 'hhh_alignment_zero_shot', 'hindi_question_answering_zero_shot', 'hindu_knowledge_zero_shot', 'hinglish_toxicity_zero_shot', 'human_organs_senses_zero_shot', 'hyperbaton_zero_shot', 'identify_math_theorems_zero_shot', 'identify_odd_metaphor_zero_shot', 'implicatures_zero_shot', 'implicit_relations_zero_shot', 'intent_recognition_zero_shot', 'international_phonetic_alphabet_nli_zero_shot', 'international_phonetic_alphabet_transliterate_zero_shot', 'intersect_geometry_zero_shot', 'irony_identification_zero_shot', 'kanji_ascii_zero_shot', 'kannada_zero_shot', 'key_value_maps_zero_shot', 'known_unknowns_zero_shot', 'language_games_zero_shot', 'language_identification_zero_shot', 'linguistic_mappings_zero_shot', 'linguistics_puzzles_zero_shot', 'list_functions_zero_shot', 'logic_grid_puzzle_zero_shot', 'logical_args_zero_shot', 'logical_deduction_zero_shot', 'logical_fallacy_detection_zero_shot', 'logical_sequence_zero_shot', 'mathematical_induction_zero_shot', 'matrixshapes_zero_shot', 'metaphor_boolean_zero_shot', 'metaphor_understanding_zero_shot', 'minute_mysteries_qa_zero_shot', 'misconceptions_russian_zero_shot', 'misconceptions_zero_shot', 'mnist_ascii_zero_shot', 'modified_arithmetic_zero_shot', 'moral_permissibility_zero_shot', 'movie_dialog_same_or_different_zero_shot', 'movie_recommendation_zero_shot', 'mult_data_wrangling_zero_shot', 'multiemo_zero_shot', 'natural_instructions_zero_shot', 'navigate_zero_shot', 'nonsense_words_grammar_zero_shot', 'novel_concepts_zero_shot', 'object_counting_zero_shot', 'odd_one_out_zero_shot', 'operators_zero_shot', 'paragraph_segmentation_zero_shot', 'parsinlu_qa_zero_shot', 'parsinlu_reading_comprehension_zero_shot', 'penguins_in_a_table_zero_shot', 'periodic_elements_zero_shot', 'persian_idioms_zero_shot', 'phrase_relatedness_zero_shot', 'physical_intuition_zero_shot', 'physics_questions_zero_shot', 'physics_zero_shot', 'play_dialog_same_or_different_zero_shot', 'polish_sequence_labeling_zero_shot', 'presuppositions_as_nli_zero_shot', 'qa_wikidata_zero_shot', 'question_selection_zero_shot', 'real_or_fake_text_zero_shot', 'reasoning_about_colored_objects_zero_shot', 'repeat_copy_logic_zero_shot', 'rephrase_zero_shot', 'riddle_sense_zero_shot', 'ruin_names_zero_shot', 'salient_translation_error_detection_zero_shot', 'scientific_press_release_zero_shot', 'semantic_parsing_in_context_sparc_zero_shot', 'semantic_parsing_spider_zero_shot', 'sentence_ambiguity_zero_shot', 'similarities_abstraction_zero_shot', 'simp_turing_concept_zero_shot', 'simple_arithmetic_json_multiple_choice_zero_shot', 'simple_arithmetic_json_subtasks_zero_shot', 'simple_arithmetic_json_zero_shot', 'simple_arithmetic_multiple_targets_json_zero_shot', 'simple_ethical_questions_zero_shot', 'simple_text_editing_zero_shot', 'snarks_zero_shot', 'social_iqa_zero_shot', 'social_support_zero_shot', 'sports_understanding_zero_shot', 'strange_stories_zero_shot', 'strategyqa_zero_shot', 'sufficient_information_zero_shot', 'suicide_risk_zero_shot', 'swahili_english_proverbs_zero_shot', 'swedish_to_german_proverbs_zero_shot', 'symbol_interpretation_zero_shot', 'temporal_sequences_zero_shot', 'tense_zero_shot', 'timedial_zero_shot', 'topical_chat_zero_shot', 'tracking_shuffled_objects_zero_shot', 'understanding_fables_zero_shot', 'undo_permutation_zero_shot', 'unit_conversion_zero_shot', 'unit_interpretation_zero_shot', 'unnatural_in_context_learning_zero_shot', 'vitaminc_fact_verification_zero_shot', 'what_is_the_tao_zero_shot', 'which_wiki_edit_zero_shot', 'winowhy_zero_shot', 'word_sorting_zero_shot', 'word_unscrambling_zero_shot']
total_bigbench = []
for name in tqdm(domain_names):
    ds = load_dataset("hails/bigbench", name, split='default')
    subset = ds.shuffle(seed=42).select(range(min(100, len(ds))))

    total_bigbench.extend(subset['inputs'])
random.shuffle(total_bigbench)
total_data.extend(total_bigbench[:length_of_dataset])

random.shuffle(total_data)
with open('benchmark_data.jsonl', 'w', encoding='utf-8') as f:
    for data in tqdm(total_data):
        f.write(json.dumps({'text': data}, ensure_ascii=False) + '\n')