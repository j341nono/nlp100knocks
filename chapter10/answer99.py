import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_id = "OuteAI/Lite-Oute-1-300M"
train_path = "data/SST-2/train.tsv"
valid_path = "data/SST-2/dev.tsv"


class PreferenceDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._prepare_preference_data(data_path)
    
    def _prepare_preference_data(self, data_path: str) -> List[Dict]:
        df = pd.read_csv(data_path, sep="\t", header=0)
        preference_pairs = []
        
        for _, row in df.iterrows():
            text = row["sentence"]
            true_label = row["label"]
            
            prompt = self._make_prompt(text)
            
            # 正解応答（望ましい）
            correct_response = "positive" if true_label == 1 else "negative"
            
            # 不正解応答（望ましくない）
            incorrect_response = "negative" if true_label == 1 else "positive"
            
            preference_pairs.append({
                'prompt': prompt,
                'chosen': prompt + " " + correct_response,
                'rejected': prompt + " " + incorrect_response,
                'true_label': true_label
            })
        
        return preference_pairs
    
    def _make_prompt(self, text: str) -> str:
        return f"""Classify the sentiment of the following text as positive or negative. Output only "positive" or "negative".
Text: {text}
Sentiment:"""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt_tokens = self.tokenizer(
            item['prompt'], 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length
        )
        
        chosen_tokens = self.tokenizer(
            item['chosen'], 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length
        )
        
        rejected_tokens = self.tokenizer(
            item['rejected'], 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length
        )
        
        return {
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(),
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
            'true_label': item['true_label']
        }

class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta: float = 0.1):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        
        # 参照モデルは勾配計算を無効化
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # シフトしたログ確率を計算
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # パディングトークンをマスク
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = selected_log_probs * mask
        
        return masked_log_probs.sum(dim=-1) / mask.sum(dim=-1)
    
    def dpo_loss(self, batch):
        # 現在のモデルのログ確率
        chosen_log_probs = self.compute_log_probs(
            self.model, batch['chosen_input_ids'], batch['chosen_attention_mask']
        )
        rejected_log_probs = self.compute_log_probs(
            self.model, batch['rejected_input_ids'], batch['rejected_attention_mask']
        )
        
        with torch.no_grad():
            ref_chosen_log_probs = self.compute_log_probs(
                self.ref_model, batch['chosen_input_ids'], batch['chosen_attention_mask']
            )
            ref_rejected_log_probs = self.compute_log_probs(
                self.ref_model, batch['rejected_input_ids'], batch['rejected_attention_mask']
            )
        
        chosen_rewards = self.beta * (chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = self.beta * (rejected_log_probs - ref_rejected_log_probs)
        
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return loss, chosen_rewards.mean(), rejected_rewards.mean()


class PPOTrainer:
    def __init__(self, model, ref_model, reward_model, tokenizer, 
                 clip_epsilon: float = 0.2, kl_penalty: float = 0.01):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        
        # 参照モデルは勾配計算を無効化
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_rewards(self, responses, true_labels):
        """報酬を計算（正解なら+1、不正解なら-1）"""
        rewards = []
        for response, true_label in zip(responses, true_labels):
            # 生成されたテキストから感情を抽出
            generated_text = self.tokenizer.decode(response, skip_special_tokens=True).lower()
            predicted_positive = "positive" in generated_text
            is_correct = (predicted_positive and true_label == 1) or (not predicted_positive and true_label == 0)
            reward = 1.0 if is_correct else -1.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
    
    def ppo_loss(self, batch, old_log_probs, rewards):
        # 現在のポリシーのログ確率
        current_log_probs = self.compute_log_probs(
            self.model, batch['chosen_input_ids'], batch['chosen_attention_mask']
        )
        
        # 重要度比
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # クリップされた目的関数
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        
        # KLペナルティ
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.ref_model, batch['chosen_input_ids'], batch['chosen_attention_mask']
            )
        kl_divergence = (current_log_probs - ref_log_probs).mean()
        
        total_loss = policy_loss + self.kl_penalty * kl_divergence
        
        return total_loss, policy_loss, kl_divergence


def train_dpo(model, tokenizer, train_dataloader, val_dataloader, num_epochs: int = 3, lr: float = 1e-5):
    """DPOトレーニング"""
    logger.info("Starting DPO training...")
    
    # 参照モデルをコピー
    ref_model = AutoModelForCausalLM.from_pretrained(model.config._name_or_path)
    ref_model.eval()
    
    trainer = DPOTrainer(model, ref_model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            loss, chosen_reward, rejected_reward = trainer.dpo_loss(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'chosen_reward': f'{chosen_reward.item():.4f}',
                'rejected_reward': f'{rejected_reward.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 評価
        if val_dataloader:
            evaluate_model(model, tokenizer, val_dataloader)


def evaluate_model(model, tokenizer, val_dataloader):
    """モデルを評価"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            for i in range(len(batch['true_label'])):
                prompt_ids = batch['prompt_input_ids'][i:i+1]
                prompt_mask = batch['prompt_attention_mask'][i:i+1]
                true_label = batch['true_label'][i].item()
                
                outputs = model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                predicted_label = 1 if "positive" in generated_text else 0
                
                if predicted_label == true_label:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    model.train()
    return accuracy


def main():
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Preparing datasets...")
    train_dataset = PreferenceDataset(train_path, tokenizer)
    val_dataset = PreferenceDataset(valid_path, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 初期評価
    logger.info("Initial evaluation:")
    initial_accuracy = evaluate_model(model, tokenizer, val_dataloader)
    
    train_dpo(model, tokenizer, train_dataloader, val_dataloader, num_epochs=2)
    
    # 最終評価
    logger.info("Final evaluation:")
    final_accuracy = evaluate_model(model, tokenizer, val_dataloader)
    
    logger.info(f"Accuracy improvement: {initial_accuracy:.4f} -> {final_accuracy:.4f}")


if __name__ == "__main__":
    main()