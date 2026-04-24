from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import os
import torch
import re

@dataclass
class scan_result:
    label: str
    score: float
    categories: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    source: str = ""

class keyword_scanner:
    def scan_input(self, text: str) -> scan_result:
        categories: list[str] = []
        evidence: list[str] = []
        score = 0

        rules = {
            "phishing": "fraud",
            "password": "credential_theft",
            "bypass": "security_bypass",
        }
    
        for keyword, category in rules.items():
            if keyword in text.lower():
                categories.append(category)
                evidence.append(keyword)
        
        if categories:
            score = min(1, 0.25 + 0.15 * len(categories))
            label = "unsafe" if score >= 0.5 else "review"
        else:
            label = "safe"

        return scan_result(
            label=label, 
            score=score, 
            categories=categories, 
            evidence=evidence,
            source="keyword_scanner"
            )

class LlamaGuard_scanner:
    CHAT_TEMPLATE_REQUEST = """\
    {%- set categories = "
    S1: Violent Crimes. 
    S2: Non-Violent Crimes. 
    S3: Sex Crimes. 
    S4: Child Exploitation. 
    S5: Defamation. 
    S6: Specialized Advice. 
    S7: Privacy. 
    S8: Intellectual Property. 
    S9: Indiscriminate Weapons. 
    S10: Hate. 
    S11: Self-Harm. 
    S12: Sexual Content. 
    S13: Elections. 
    S14: Code Interpreter Abuse. 
    " -%}
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Task: Check if there is unsafe content in 'User' messages according to our safety policy. 
    <BEGIN UNSAFE CONTENT CATEGORIES>
    {{ categories }}
    <END UNSAFE CONTENT CATEGORIES>
    <BEGIN CONVERSATION>
    {% for message in messages %}
    {%- if message.role == 'user' %}
    User: {{ message.content }}
    {%- elif message.role == 'assistant' %}
    Agent: {{ message.content }}
    {%- endif %}
    {% endfor %}
    <END CONVERSATION>
    Provide your safety assessment for ONLY THE User message in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must be a comma-separated list of violated categories (e.g., S1, S10).
    - If safe, output only the word 'safe'.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    HF_MODEL = "meta-llama/Llama-Guard-3-8B"

    def __init__(self, api_key):
        self.__api = api_key
        login(token=os.getenv("HF_TOKEN"))
        self.__tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL, token=self.__api)
        self.__model = AutoModelForCausalLM.from_pretrained(self.HF_MODEL, dtype=torch.float16, device_map="auto", token=self.__api)

    def _prepare_input_request(self, prompt: str, reponse: str) -> list[dict[str, str]]:
        chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reponse}]
        template_chat = self.__tokenizer.apply_chat_template(chat, tokenize = False, chat_template = self.CHAT_TEMPLATE_REQUEST)
        return [{"role": "user", "content": template_chat}]
    
    def _generate(self, input: list[dict]) -> str:
        text = input[0]["content"]
        tokenized = self.__tokenizer(text, return_tensors="pt").to("cuda")
        input_len = tokenized["input_ids"].shape[-1]
        with torch.no_grad():
            generated = self.__model.generate(**tokenized, max_new_tokens=100, do_sample=False)
        output = self.__tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
        return output
    
    def classify_request(self, prompt: str, response: str) -> scan_result:
        input_request = self._prepare_input_request(prompt, response)
        output = self._generate(input_request)
        content = output.strip().lower()
        print(content)
        if "unsafe" not in content:
            return scan_result(
            label="safe", 
            score=0, 
            source="LlamaGuard_scanner"
            )
        else:
            risk_list = re.findall(r"S(\d+)", content)
            return scan_result(
            label="unsafe", 
            score=min(1, 0.25 + 0.15 * len(risk_list)), 
            categories=['S' + r for r in risk_list], 
            source="LlamaGuard_scanner"
            )