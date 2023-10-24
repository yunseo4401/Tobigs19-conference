import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments, AutoModelForSequenceClassification
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.preprocess import load, split_and_tokenize
import wandb



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    


if __name__ == '__main__':

    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser()

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--maxlen', type=int, default=512)
    parser.add_argument('--model_name', type=str, default="monologg/koelectra-small-v2-distilled-korquad-384")
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_initial', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # args 에 위의 내용 저장
    args = parser.parse_args()



    

    if args.model_name == 'skt/kobert-base-v1':
        from kobert_tokenizer import KoBERTTokenizer
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    # if args.model_name == 'monologg/kobert':
    #     from kobert_tokenizer import KoBERTTokenizer
    #     tokenizer = KoBERTTokenizer.from_pretrained('monologg/kobert')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config, resume_download=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 데이터 준비
    texts, labels = load(tokenizer)
    train_dataset, val_dataset, test_dataset = split_and_tokenize(texts, labels, tokenizer, maxlen = args.maxlen)

    # 실험 설정
    train_args = TrainingArguments(
        output_dir=f'./results-{args.model_name}-{args.epoch}ep-{args.batch_size}bs',            # output directory
        learning_rate=args.lr_initial,
        num_train_epochs=args.epoch,        # total number of training epochs
        per_device_train_batch_size=args.batch_size,     # batch size per device during training
        per_device_eval_batch_size=64,      # batch size for evaluation
        warmup_steps=args.warmup_steps,                    # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,                  # strength of weight decay
        logging_steps=10,
        load_best_model_at_end = True,
        save_strategy = "no",
        metric_for_best_model = 'f1',
        report_to="wandb"
    )
    print(train_args)

    trainer = Trainer(
        model=model,                        # the instantiated 🤗 Transformers model to be trained
        args=train_args,                    # training arguments, defined above
        train_dataset=train_dataset,        # training dataset
        eval_dataset=val_dataset,           # evaluation dataset
        compute_metrics = compute_metrics
    )
    

    print('--------------------------training-------------------------------')

    wandb.init(project='AdArticleClassification', entity='the-huistle', tags=['plotConfusionMatrix', 'KoBERT'], name=f'{args.model_name}-{args.epoch}epoch-{args.batch_size}batch-{args.weight_decay}decay')

    trainer.train()
    
    # save model
    torch.save(model.state_dict(), f'./{args.model_name}-{args.epoch}ep-{args.batch_size}bs.pt')
    # trainer.save_model(f'{args.model_name}-{args.epoch}ep-{args.batch_size}bs.pt')
    
    model.eval()

    val_preds = trainer.predict(val_dataset)
    logits = val_preds.predictions
    logits = torch.from_numpy(logits)

    scores = trainer.evaluate(eval_dataset=val_dataset)

    wandb.log({'ROC curve': wandb.plots.ROC(y_true=val_dataset.labels, y_probas=list(logits.softmax(dim=-1).numpy()), labels=list(val_preds.label_ids)),
               'Confusion Matrix': wandb.plot.confusion_matrix(probs=None, y_true=val_dataset.labels, preds=list(val_preds.label_ids), class_names=['일반기사', '광고성기사']),
               'accuracy': scores['eval_accuracy'],
               'f1 score': scores['eval_f1'],
               'precision': scores['eval_precision'],
               'recall': scores['eval_recall']})

    wandb.finish()

