import string
import re
from rouge import Rouge
from collections import Counter
from konlpy.tag import Mecab
import datasets

from sklearn.metrics import recall_score, precision_score,f1_score,accuracy_score

def clean_up(text):
    text =text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text   

def clean(text):
    REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9가-힣]")
    return REMOVE_CHAR_PATTERN.sub(" ", text.lower()).strip()

def metric_rouge_korean(preds, refs):    
    mecab = Mecab()
    metric = datasets.load_metric('rouge')

    predictions = []
    references = []
    predictions += [" ".join(mecab.morphs(clean(pred))) for pred in preds]
    references += [" ".join(mecab.morphs(clean(ref))) for ref in refs]
    #print(predictions)
    #print(references)
    results = metric.compute(predictions=predictions, references=references)
    #print(results)
    #exit()
    #print(f"[ROUGE-L] {results['rougeL'].mid} \n[ROUGE-1] {results['rouge1'].mid} \n[ROUGE-2] {results['rouge2'].mid}")
    
    return {'rouge': results['rougeL'].mid.fmeasure}



def metric_rouge_english(preds, refs):
    metric = datasets.load_metric('rouge')
    
    results = metric.compute(predictions=preds, references=refs)
    
    #print(f"[ROUGE-L] {results['rougeL'].mid} \n[ROUGE-1] {results['rouge1'].mid} \n[ROUGE-2] {results['rouge2'].mid}")
    return {'rouge': results['rougeL'].mid.fmeasure}

def metric_bleu_korean(preds, refs):
    mecab = Mecab()
    metric = datasets.load_metric('bleu')

    predictions = []
    references = []
    for pred in preds:
        prediction = (" ".join(mecab.morphs(clean(pred)))).split(' ')
        predictions += [prediction]
    for ref in refs:
        reference = (" ".join(mecab.morphs(clean(ref)))).split(' ')
        references += [reference]
    references = [[e] for e in references]
    
    results = metric.compute(predictions=predictions, references=references)
    
    print(f"[BLEUScore] {results['bleu']:.3f}, Precisions: {results['precisions']}")

def metric_bleu_english(preds, refs):

    metric = datasets.load_metric('bleu')

    predictions = []
    references = []

    for pred in preds:
        prediction = (" ".join(clean(pred))).split(' ')
        predictions += [prediction]
    for ref in refs:
        reference = (" ".join(clean(ref))).split(' ')
        references += [reference]
    references = [[e] for e in references]
    
    results = metric.compute(predictions=predictions, references=references)
    
    print(f"[BLEUScore] {results['bleu']:.3f}, Precisions: {results['precisions']}")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return (white_space_fix(remove_punc(lower(s))))

def accuracy_match_score_normalize(prediction, ground_truth):
    if normalize_answer(prediction)== '' or normalize_answer(ground_truth)== '':
        return 0
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_match_score(prediction, ground_truth):
    return accuracy_match_score_normalize(prediction, ground_truth)

def accuracy_match_score(prediction, ground_truth):
    return int(prediction.strip() == ground_truth.strip())

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def calculate_bleu_scores(predictions, ground_truths):
    pass

def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_accuracy_scores(predictions, ground_truths):
    accuracy = 0
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        accuracy += accuracy_match_score(prediction, ground_truth)
    accuracy /= len(predictions)
    return accuracy*100

def calculate_em_scores(predictions, ground_truths):
    em = 0
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        em += exact_match_score(prediction, ground_truth)
    em /= len(predictions)
    return em*100

def calculate_f1_scores(predictions, ground_truths, ids=None):
    f1_score = 0 
    for i in range(len(predictions)):
        ground_truth = ground_truths[i]
        prediction = predictions[i]
        f1_score += _f1_score(prediction, ground_truth)

    f1_score /= len(predictions)
    return f1_score*100

def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return lmap(str.strip, gen_text)

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))

def generation_metric(preds, refs, target_language, eval=False):
    if target_language == 'ko':
        rouge_score = metric_rouge_korean(preds, refs)
        metric_bleu_korean(preds, refs)
    elif target_language == 'en':
        rouge_score = metric_rouge_english(preds, refs)
        metric_bleu_english(preds, refs)
    
    if eval==True:
        print(f"ROUGE-L Score : {rouge_score}")
        accuracy = calculate_accuracy_scores(preds, refs)
        print(f"Accuracy : {accuracy}")
        return rouge_score