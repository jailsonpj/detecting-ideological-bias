import torch
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)

class LLMShortLearning:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_dtype = getattr(torch, "float16")
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=True, 
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            cache_dir="/home/bolsista/jailson-mestrado/llm_experimentos/llama-hf", 
            torch_dtype=self.compute_dtype,
            quantization_config=self.bnb_config,
            device_map=self.device
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            cache_dir="/home/bolsista/jailson-mestrado/llm_experimentos/llama-hf",
            trust_remote_code=True,
            padding_side="left",
            pad_token="[PAD]"
        )
        self.model = self.model.bfloat16()

    def get_model_tokenizer(self):
        model, tokenizer = setup_chat_format(self.model, self.tokenizer)
        return model, tokenizer