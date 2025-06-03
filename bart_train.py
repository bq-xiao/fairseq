import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from fairseq import tasks, optim

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm",
                                          cache_dir="download_model/chinese-bert-wwm",
                                          local_files_only = True)

cfg = OmegaConf.load("config.yaml")
task = tasks.setup_task(cfg.task)
#task = TranslationTask(cfg, tokenizer.vocab, tokenizer.vocab)
model = task.build_model(cfg.model)
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

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=4)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=4)
criterion = task.build_criterion(cfg.criterion)
optimizer = optim.build_optimizer(cfg.optimizer, model.parameters())
_num_updates = 0
for epoch in range(10):
    model.train()
    criterion.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_dataloader):
        batch = {
            "ntokens": data['src_lengths'].sum().item(),
            "net_input": {
                "src_tokens": data['input_ids'],
                "src_lengths": data['src_lengths'],
                'prev_output_tokens': data['labels']
            },
            "target": data['labels'],
        }
        loss, sample_size_i, logging_output = task.train_step(
            sample=batch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            update_num=_num_updates,
            ignore_grad=False,
        )
        _num_updates = _num_updates + 1
        print(f"train loss:{loss.item()}")
        #print(logging_output)
    with torch.no_grad():
        model.eval()
        criterion.eval()
        valid_losses = []
        for i, data in enumerate(test_dataloader):
            batch = {
                "ntokens": data['src_lengths'].sum().item(),
                "net_input": {
                    "src_tokens": data['input_ids'],
                    "src_lengths": data['src_lengths'],
                    'prev_output_tokens': data['labels']
                },
                "target": data['labels'],
            }
            _loss, sample_size, logging_output = task.valid_step(batch, model, criterion)
            print(f"valid loss:{_loss.item()}")

    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "bart_translation.pt")