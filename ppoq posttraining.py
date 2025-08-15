# ppo_with_quantum_critic.py
#! -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import sys
sys.path.append("./PPO-Q")
from model.utils import INIT_METHOD, orthogonal_init
import torchquantum as tq  # noqa: F401
from model.models import PQCLayer  # pragma: no cover

# -----------------------------
# Dataset helper (simple)
# -----------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []
        for prompt in prompts:
            if apply_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                # bos token if presentadd_t
                bos = getattr(self.tokenizer, "bos_token", "")
                prompt = bos + prompt
            self.final_prompts.append(prompt)

    def __len__(self):
        return len(self.final_prompts)

    def __getitem__(self, index):
        return self.final_prompts[index]

# -----------------------------
# QuantumCritic (integrated)
# -----------------------------
class QuantumCritic(nn.Module):
    """
    QuantumCritic: hidden -> pre_encoding_net (hidden_size -> n_wires)
                   -> pqc_layer (n_wires)
                   -> post_processing_net (n_wires -> 1)
    forward returns (batch, num_actions)
    """
    def __init__(self,
                 base_model,
                 n_wires: int = 6,
                 n_blocks: int = 2,
                 input_dim: int = None,
                 output_dim: int = 1,
                 ini_method=("X", "O"),
                 use_quafu: bool = False,
                 use_quafu_simulator: bool = True,
                 quantum_device: str = 'Dongling'):
        super().__init__()
        self.base_model = base_model
        # We'll use base_model to extract hidden states. By default do not freeze base_model's params.
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = True

        if input_dim is None:
            input_dim = getattr(base_model.config, "hidden_size", None)
            if input_dim is None:
                raise ValueError("请提供 input_dim 或 确保 base_model.config.hidden_size 可用")

        # pre-encoding linear layer
        if ini_method[0] != "NOT":
            self.pre_encoding_net = nn.Linear(input_dim, n_wires)
            try:
                INIT_METHOD[ini_method[0]](self.pre_encoding_net)
            except Exception:
                pass
        else:
            if n_wires != input_dim:
                n_wires = input_dim
            self.pre_encoding_net = nn.Identity()

        # PQC (or fallback)
        self.pqc_layer = PQCLayer(n_wires=n_wires, n_blocks=n_blocks,
                                  use_quafu=use_quafu, use_quafu_simulator=use_quafu_simulator,
                                  quantum_device=quantum_device)

        # post processing
        self.post_processing_net = nn.Linear(n_wires, output_dim)
        if 'O' == ini_method[-1]:
            try:
                INIT_METHOD[ini_method[-1]](self.post_processing_net, 0.01)
            except Exception:
                pass
        else:
            try:
                INIT_METHOD[ini_method[-1]](self.post_processing_net)
            except Exception:
                pass

        self.is_critic = True

    def forward(self, input_ids, attention_mask, num_actions):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        num_actions: int
        returns: values (batch, num_actions)
        """
        device = next(self.post_processing_net.parameters()).device

        # get hidden states from base model
        # NOTE: using no_grad here to avoid huge memory use; if you want to finetune base, remove no_grad.
        with torch.no_grad():
            out = self.base_model(input_ids.to(device), attention_mask=attention_mask.to(device))
            hidden_state = out.last_hidden_state  # (B, seq_len, hidden_size)

        if isinstance(num_actions, torch.Tensor):
            num_actions = int(num_actions.item())
        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        seq_len = hidden_state.size(1)
        if seq_len < num_actions:
            # pad front if too short (rare)
            pad_len = num_actions - seq_len
            pad_tensor = torch.zeros(hidden_state.size(0), pad_len, hidden_state.size(2), device=hidden_state.device, dtype=hidden_state.dtype)
            selected = torch.cat([pad_tensor, hidden_state], dim=1)[:, -num_actions:, :]
        else:
            selected = hidden_state[:, -num_actions:, :]  # (B, T, H)

        B = selected.size(0)
        BT = B * num_actions
        x_flat = selected.contiguous().view(BT, -1)  # (BT, H)

        # pre-encoding to n_wires
        x_enc = self.pre_encoding_net(x_flat.to(device))  # (BT, n_wires)

        # PQC
        q_out = self.pqc_layer(x_enc)  # (BT, n_wires)

        # post processing to scalar
        v_flat = self.post_processing_net(q_out)  # (BT, 1) or (BT,)
        if v_flat.dim() == 2 and v_flat.size(-1) == 1:
            v_flat = v_flat.squeeze(-1)

        # reshape back to (B, T)
        values = v_flat.view(B, num_actions)

        return values

# -----------------------------
# The rest: PPO helpers (kept largely your original logic)
# -----------------------------
def compute_policy_loss(log_probs, old_log_probs, advantages, action_mask=None, clip_eps=0.2):
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2)
    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


def compute_value_loss(values, old_values, returns, action_mask=None, clip_eps: float = None):
    if clip_eps is not None:
        values_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
    else:
        loss = (values - returns) ** 2

    if action_mask is None:
        return loss.mean(-1).mean()
    return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()


class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences):
        batch = [{} for _ in range(len(experiences))]
        keys = (
            "seqs",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
            "num_actions"
        )
        for key in keys:
            for i, experience in enumerate(experiences):
                value = getattr(experience, key)
                batch[i][key] = value

        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit:
            self.buffer = self.buffer[len(self.buffer) - self.limit:]

    def get_batches(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]


@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    reward: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor
    num_actions: Union[int, torch.Tensor]
    kl: Optional[torch.Tensor] = None


def compute_approx_kl(
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
):
    log_ratio = log_probs.float() - ref_log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio


def get_advantages_and_returns(
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float):
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


def generate_samples(prompts, model, max_length, max_new_tokens, n_samples_per_prompt, micro_rollout_batch_size):
    samples_list = []
    model.eval()
    all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in prompts], [])
    for i in range(0, len(all_prompts), micro_rollout_batch_size):
        batch_prompts = all_prompts[i:i + micro_rollout_batch_size]
        inputs = actor_tokenizer(batch_prompts, padding='max_length', max_length=max_length, truncation=True,
                                 return_tensors='pt')
        input_ids = inputs['input_ids']
        seqs = model.generate(**inputs.to(device),
                              max_new_tokens=max_new_tokens,
                              eos_token_id=eos_token_id,
                              pad_token_id=pad_token_id)
        if seqs.size(1) >= max_new_tokens + max_length:
            seqs = seqs[:, :max_new_tokens + max_length]
        else:
            seqs = torch.cat([seqs, torch.full((seqs.size(0), max_new_tokens + max_length - seqs.size(1)),
                                               fill_value=pad_token_id, device=seqs.device)], dim=1)

        attention_mask = (seqs.ne(pad_token_id)).to(dtype=torch.long)
        ans = seqs[:, input_ids.size(1):]
        action_mask = (ans.ne(eos_token_id) & ans.ne(pad_token_id)).to(dtype=torch.long)

        samples = Samples(
            seqs=seqs,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)

    return samples_list


def compute_rewards(kl, r, action_mask, kl_ctl, clip_reward_value):
    kl_divergence_estimate = -kl_ctl * kl
    rewards = kl_divergence_estimate

    ends = action_mask.sum(1) + 1

    if not isinstance(clip_reward_value, torch.Tensor):
        clip_reward_value = torch.tensor(clip_reward_value).to(r.device)

    reward_clip = torch.clamp(r, -clip_reward_value,
                              clip_reward_value)
    batch_size = r.size(0)
    for j in range(batch_size):
        rewards[j, :ends[j]][-1] += reward_clip[j, 0]

    return rewards


def generate_experiences(samples_list):
    actor_model.eval()
    ref_model.eval()
    reward_model.eval()
    critic_model.eval()

    experiences = []
    for samples in samples_list:
        seqs = samples.seqs
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        with torch.no_grad():
            output = actor_model(seqs, attention_mask=attention_mask)
            logits = output.logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

            ref_output = ref_model(seqs, attention_mask=attention_mask)
            ref_logits = ref_output.logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_log_probs_labels = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1))
            ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]

            # value from quantum critic
            value = critic_model.forward(seqs, attention_mask, num_actions).to(device)

            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_model_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True)
            r = reward_model(**reward_model_inputs.to(device)).logits

            kl = compute_approx_kl(
                action_log_probs,
                ref_action_log_probs,
                action_mask=action_mask).to(device)

            rewards = compute_rewards(kl, r, action_mask, kl_ctl=0.1, clip_reward_value=0.2)
            advantages, returns = get_advantages_and_returns(value, rewards, action_mask, gamma=0.1, lambd=0.2)

        experiences.append(Experience(seqs,
                                      action_log_probs.detach(),
                                      value.detach(),
                                      returns.detach(),
                                      advantages.detach(),
                                      attention_mask,
                                      action_mask,
                                      r.detach(),
                                      samples.response_length,
                                      samples.total_length,
                                      num_actions,
                                      kl.detach(),
                                      ))
    return experiences


@dataclass
class BufferItem:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: Union[int, torch.Tensor]


def collate_fn(batch):
    seqs = []
    action_log_probs = []
    values = []
    returns = []
    advantages = []
    attention_mask = []
    action_mask = []

    for x in batch:
        seqs.append(x['seqs'])
        action_log_probs.append(x['action_log_probs'])
        values.append(x['values'])
        returns.append(x['returns'])
        advantages.append(x['advantages'])
        attention_mask.append(x['attention_mask'])
        action_mask.append(x['action_mask'])

    seqs = torch.cat(seqs, dim=0)
    action_log_probs = torch.cat(action_log_probs, dim=0)
    values = torch.cat(values, dim=0)
    returns = torch.cat(returns, dim=0)
    advantages = torch.cat(advantages, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    action_mask = torch.cat(action_mask, dim=0)

    return BufferItem(seqs, action_log_probs, values, returns, advantages, attention_mask, action_mask,
                      action_mask.size(1))


def train_step(experience, steps):
    actor_model.train()
    optimizer_actor.zero_grad()

    sequences = experience.seqs
    old_action_log_probs = experience.action_log_probs
    advantages = experience.advantages
    num_actions = experience.num_actions
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    old_values = experience.values
    returns = experience.returns

    logits = actor_model(sequences, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=sequences[:, 1:].unsqueeze(-1))
    action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]

    policy_loss = compute_policy_loss(action_log_probs, old_action_log_probs, advantages, action_mask=action_mask)
    policy_loss.backward()
    optimizer_actor.step()
    writer.add_scalar("policy_loss", policy_loss.item(), steps)

    critic_model.train()
    optimizer_critic.zero_grad()
    values = critic_model.forward(sequences, attention_mask, num_actions)
    value_loss = compute_value_loss(values, old_values, returns, action_mask)
    value_loss.backward()
    optimizer_critic.step()
    writer.add_scalar("value_loss", value_loss.item(), steps)
    print(f"step: {steps}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")


def train():
    buffer = ExperienceBuffer(limit=100)
    steps = 0
    for episode in range(episodes):
        for rand_prompts in prompts_dataloader:
            samples = generate_samples(rand_prompts, actor_model, max_length, max_new_tokens, n_samples_per_prompt,
                                       micro_rollout_batch_size)
            experiences = generate_experiences(samples)
            buffer.append(experiences)
            dataloader = DataLoader(buffer, batch_size=micro_train_batch_size, shuffle=True, collate_fn=collate_fn)
            torch.cuda.empty_cache()
            for epoch in range(max_epochs):
                for experience in dataloader:
                    train_step(experience, steps)
                    steps += 1

            buffer.clear()
            torch.cuda.empty_cache()


# -----------------------------
# Main: initialize models, tokenizers, dataset
# -----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparams (you can tune)
    episodes = 3
    max_epochs = 5
    rollout_batch_size = 8
    micro_rollout_batch_size = 2
    n_samples_per_prompt = 2
    max_new_tokens = 50
    max_length = 256
    micro_train_batch_size = 2

    writer = SummaryWriter('./runs')

    # Replace these paths with your model names or local paths
    actor_model = AutoModelForCausalLM.from_pretrained('qwen-2.5-1.5b').to(device)
    ref_model = AutoModelForCausalLM.from_pretrained('qwen-2.5-1.5b').to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained('reward-model-deberta-v3-large-v2').to(device)
    actor_tokenizer = AutoTokenizer.from_pretrained('qwen-2.5-1.5b')
    reward_tokenizer = AutoTokenizer.from_pretrained('reward-model-deberta-v3-large-v2')
    # Use our QuantumCritic (n_wires = 6)
    critic_model = QuantumCritic(actor_model.base_model, n_wires=6, n_blocks=2, input_dim=actor_model.config.hidden_size).to(device)
    optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=5e-5)
    optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=5e-5)
    actor_tokenizer.padding_side = 'left'
    eos_token_id = actor_tokenizer.eos_token_id
    pad_token_id = actor_tokenizer.pad_token_id

    prompt_list = [
        '请问1+1等于多少？',
        'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
        '为什么人们喜欢在水族馆里游泳，而不是在游泳池里？',
        '你是一位营销专家。为Instagram reels写30个带有营销技巧的脚本。',
        '为什么所有的镜子都是矩形的？',
        '我们在受感染的植物根部可以找到哪一种，臭氧还是金子？'
    ]
    prompts_dataset = PromptDataset(prompt_list, actor_tokenizer, apply_chat_template=True)
    prompts_dataloader = DataLoader(prompts_dataset, batch_size=rollout_batch_size, shuffle=True)

    train()
