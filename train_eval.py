import torch
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from typing import *


def train_model(
    epochs: int, model: torch.nn.Module, 
    tokenizer: transformers.AutoTokenizer, loaders: List[DataLoader], 
    optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
    device: str='cpu'
) -> torch.nn.Module:
    # cross entropy loss
    loss_function1 = torch.nn.CrossEntropyLoss()
    loss_function2 = torch.nn.CrossEntropyLoss()
    
    # извлечение DataLoaders
    if len(loaders) > 1:
        train_loader = loaders['train']
        val_loader = loaders['val']
        steps_per_epoch = [('train', train_loader), ('val', val_loader)]
    else:
        train_loader = loaders['train']
        steps_per_epoch = [('train', train_loader)]

    # обучение по эпохам
    for epoch in range(epochs):
        for mode, loader in steps_per_epoch:
            # сохранение статистик
            train_loss = 0
            n_correct = 0
            processed_data = 0
            
            # train/val 
            if mode == 'train':
                model.train()
                requires_grad_mode = True
            else:
                model.eval()
                requires_grad_mode = False
            
            # проход по батчам
            for data in tqdm(loader):
                # обнуляем градиенты
                optimizer.zero_grad()

                # извлечение входных данных для модели
                inputs = tokenizer(
                    data['text'], padding=True, truncation=True, 
                    add_special_tokens=True, return_tensors='pt'
                )
                ids = inputs['input_ids'].to(device)
                mask = inputs['attention_mask'].to(device)
                token_type_ids = inputs["token_type_ids"].to(device)
                target1 = data['target1'].to(device)
                target2 = data['target2'].to(device)
                
                # устанавливаем необходимость вычислять/не_вычислять градиенты
                with torch.set_grad_enabled(requires_grad_mode):
                    outputs = model(ids, mask, token_type_ids)
                    preds = torch.argmax(outputs.data, dim=1)

                    # настраиваем модели на конкретный target
                    if all(target1 == target2):
                        loss1 = loss_function1(outputs, target1)
                        train_loss += loss1.item() * outputs.size(0)
                        n_correct += torch.sum(preds == target1)
                        if mode == 'train':
                            # вычисляем градиенты и обновляем веса
                            loss1.backward()
                            optimizer.step()
                    # если у твита более чем 1 метка, то настраиваем на обе
                    else:
                        loss1 = loss_function1(outputs, target1) * 0.5
                        loss2 = loss_function2(outputs, target2) * 0.5
                        loss_all = loss1 + loss2
                        train_loss += loss_all.item() * outputs.size(0)

                        mask_singular = target1 == target2
                        mask_multiple = target1 != target2
                        singular = preds[mask_singular]
                        n_correct += torch.sum(singular == target1[mask_singular])
                        multiple = preds[mask_multiple]
                        n_correct += torch.sum(
                            (multiple == target1[mask_multiple]) & \
                            (multiple == target2[mask_multiple])
                        )
                        if mode == 'train':
                            # вычисляем градиенты и обновляем веса
                            loss_all.backward()
                            optimizer.step()     
                    processed_data += outputs.size(0)

            # вычисляем ошибку и точность прогноза на эпохе
            loader_loss = train_loss / processed_data
            loader_acc = n_correct.cpu().numpy() / processed_data
            print(f'{epoch + 1} epoch with {mode} mode has: {loader_loss} loss, {loader_acc} acc')
        
        # делаем шаг для sheduler оптимайзера
        scheduler.step()

    return model


def calculate_accuracy(
    model: torch.nn.Module, SentimentData: Dataset,
    device: str='cpu'
) -> float:
    model.eval()
    loader = DataLoader(SentimentData, batch_size=10, shuffle=False)
    n_correct = 0
    processed_data = 0
    
    for data in tqdm(loader):
        inputs = model.tokenizer(
            data['text'], padding=True, 
            add_special_tokens=True, return_tensors='pt'
        )
        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        target1 = data['target1'].to(device)
        target2 = data['target2'].to(device)
        
        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)
            preds = torch.argmax(outputs.data, dim=1)
            mask_singular = target1 == target2
            mask_multiple = target1 != target2
            singular = preds[mask_singular]
            n_correct += torch.sum(singular == target1[mask_singular])
            multiple = preds[mask_multiple]
            if len(multiple) > 0:
                n_correct += torch.sum((multiple == target1[mask_multiple]) & (multiple == target2[mask_multiple]))
            processed_data += outputs.size(0)
        
    loader_acc = n_correct.cpu().numpy() / processed_data
    
    return loader_acc


def calculate_f1_class(
    model: torch.nn.Module, SentimentData: Dataset, 
    class_num: int, device: str='cpu'
) -> float:
    model.eval()
    loader = DataLoader(SentimentData, batch_size=10, shuffle=False)
    true_positive = 0
    false_positive, false_negative = 0, 0
    
    for data in tqdm(loader):
        inputs = model.tokenizer(
            data['text'], padding=True, 
            add_special_tokens=True, return_tensors='pt'
        )
        ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        target1 = data['target1'].to(device)
        
        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)
            
            preds = torch.argmax(outputs.data, dim=1)
            preds = preds.cpu().numpy()
            target1 = target1.cpu().numpy()
            
            mask_positive = target1 == class_num
            mask_negative = target1 != class_num
            
            true_positive += np.sum(preds[mask_positive] == class_num)
            false_positive += np.sum(preds[mask_negative] == class_num)
            false_negative += np.sum(preds[mask_positive] != class_num)
        
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    loader_f1 = 2 * precision * recall / (precision + recall)
    
    return loader_f1