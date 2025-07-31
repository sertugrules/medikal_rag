import json
import evaluate
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import score as bert_score

# 1. Veriyi yükle
with open("response.json", "r", encoding="utf-8") as f:
    data = json.load(f)

predictions = [item["model_answer"] for item in data]
references = [item["ground_truth"] for item in data]

# 2. BLEU
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# 3. ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)

# 4. METEOR
meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=predictions, references=references)

# 5. BERTScore
P, R, F1 = bert_score(predictions, references, lang="en")
bert_result = {
    "bert_precision": round(P.mean().item(), 4),
    "bert_recall": round(R.mean().item(), 4),
    "bert_f1": round(F1.mean().item(), 4)
}

# 6. Perplexity (GPT-2 ile)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

pp_scores = []
for pred in tqdm(predictions, desc="Calculating Perplexity"):
    try:
        pp = calculate_perplexity(pred)
        pp_scores.append(pp)
    except Exception:
        pp_scores.append(None)

valid_pp = [p for p in pp_scores if p is not None]
average_pp = sum(valid_pp) / len(valid_pp)

print("\n--- METRICS ---")
print("BLEU:", bleu_result)
print("ROUGE:", rouge_result)
print("METEOR:", meteor_result)
print("BERTScore:", bert_result)
print(f"Average Perplexity: {average_pp:.2f}")
