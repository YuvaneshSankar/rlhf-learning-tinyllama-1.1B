import torch
from trl import PPOTrainer , PPOConfig
from unsloth import FastLanguageModel
from peft import PeftModel
from rm import get_reward_score
from torch.utils.data import DataLoader
from datasets import load_dataset


dataset = load_dataset("tatsu-lab/alpaca", split="train")
prompts = dataset['instruction']


dataloader = DataLoader(
    prompts,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda x: x
)


def load_model_and_tokenizer(max_seq_length ,model_name):
  base_model,tokenizer=FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
  )

  return base_model,tokenizer


def load_adapter(base_model, adapter_dir):
  model=PeftModel.from_pretrained(base_model, adapter_dir)
  return model


def ppo_training():
    base_model, tokenizer = load_model_and_tokenizer(2048, "unsloth/tinyllama-bnb-4bit")
    model = load_adapter(base_model, "./models/sft/checkpoint-6000")

    ref_model = load_adapter(base_model, "./models/sft/checkpoint-6000")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        num_ppo_epochs=4,
        kl_coef=0.2,
        cliprange=0.2,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )


    for epoch in range(3):
        for batch in dataloader:

            query_tensors = [tokenizer(p, return_tensors="pt").input_ids.squeeze(0) for p in batch]

            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=128,
                do_sample=True, #we will use sampling for more diverse outputs
                temperature=0.7,
                top_p=0.9
            )

            # Decode responses
            responses_text = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            rewards = [torch.tensor(get_reward_score(p, r)) for p, r in zip(batch, responses_text)]

            # PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log every 10 steps
            if ppo_trainer.current_step % 10 == 0:
                print(f"Epoch {epoch}, Step {ppo_trainer.current_step}, Reward: {torch.stack(rewards).mean():.2f}")


        model.save_pretrained(f"./models/ppo/epoch_{epoch}")



if __name__ == "__main__":
  ppo_training()
