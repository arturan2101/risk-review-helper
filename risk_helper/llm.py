from typing import Protocol

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a careful assistant that drafts risk assessments. Reply with JSON only."


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class FakeLLM:
    # Returns canned responses in order and repeats the last one, so tests can simulate
    # "bad answer, then good answer" without a real model. Prompts are recorded for asserts.
    def __init__(self, *responses: str) -> None:
        if not responses:
            raise ValueError("FakeLLM needs at least one response")
        self._responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts), len(self._responses)) - 1
        return self._responses[index]


class HuggingFaceLLM:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 300,
    ) -> None:
        # Lazy import: the core package, tests and CI must work without transformers/torch.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceLLM needs optional dependencies: pip install -r requirements-llm.txt"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        output = self._model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,  # greedy decoding: the same input always gives the same answer
        )
        # generate() returns prompt + answer; decode only the newly generated tokens.
        answer_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(answer_tokens, skip_special_tokens=True)
