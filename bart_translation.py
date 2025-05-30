import evaluate
import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from fairseq import tasks
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm", cache_dir="download_model/chinese-bert-wwm")

cfg = OmegaConf.load("config.yaml")
task = tasks.setup_task(cfg.task)
model = task.build_model(cfg.model)
model.encoder.embed_tokens = nn.Embedding(tokenizer.vocab_size, 768, padding_idx=1)
model.decoder.embed_tokens = nn.Embedding(tokenizer.vocab_size, 768, padding_idx=1)
model.decoder.output_projection = nn.Linear(in_features=768, out_features=tokenizer.vocab_size, bias=False)
print(model)

file = r'news-commentary-v18.en-zh.tsv'
lines = open(file, encoding='utf-8').read().strip().split('\n')
# Split every line into pairs and normalize
rows = []
for l in lines:
    row = l.split('\t')
    if len(row) < 2 or len(row[0]) == 0 or len(row[1]) == 0:
        continue
    rows.append({'input_text': row[0], 'target_text': row[1]})
dataset = Dataset.from_list(rows)

def preprocess_function(examples):
    inputs, targets = examples['input_text'], examples['target_text']

    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", return_tensors='pt', truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", return_tensors='pt', truncation=True)
    src_lengths = torch.LongTensor(
        [s.ne(0).long().sum() for s in model_inputs["input_ids"]]
    )
    model_inputs["src_lengths"] = src_lengths
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns= ['input_text','target_text'])
tokenized_datasets.set_format("torch")
tokenized_datasets = tokenized_datasets.train_test_split(train_size=0.8)

#loss_fn = task.build_criterion(cfg.criterion)
device = "cuda"

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=4)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=4)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0006, betas=(0.9, 0.98), eps=1e-06
)
model.to(device)
metric = evaluate.load("sacrebleu")

for epoch in range(10):
    for i, data in enumerate(train_dataloader):
        x = data['input_ids'].to(device)
        tgt = data['labels'].to(device)
        src_lengths = data['src_lengths'].to(device)
        y, _ = model(src_tokens=x, src_lengths =src_lengths, prev_output_tokens=tgt)
        #y = y.argmax(-1)
        #tgt = tgt.argmax(-1)
        optimizer.zero_grad()  # 清空梯度
        a, b = y.view(-1, tokenizer.vocab_size), tgt.view(-1)
        loss = loss_fn(a, b)   # 计算损失
        print(loss.item())
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 更新模型参数
        if i % 100 == 0:
            print(
            f"\t Train step: {i + 1}/{len(train_dataloader)} | Train step Loss: {loss.item():7.3f} | Train step PPL: {np.exp(loss.item()):7.3f}")

    preds, labels = [], []
    for batch_data in test_dataloader:
        x = batch_data['input_ids'].to(device)
        with torch.no_grad():
            y, _ = model(src_tokens=x)
        label_tokens = batch_data["labels"].cpu().numpy()
        y = y.cpu().numpy()
        decoded_preds = tokenizer.batch_decode(y, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=preds, references=labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"\t Train bleu:{result.items()}")

torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "bart_translation.pt")
