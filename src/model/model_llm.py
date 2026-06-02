import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class ChatLLM:
    def __init__(self, model, tokenizer, max_new_tokens=128, max_input_tokens=3072):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    def invoke(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý AI phân tích tài liệu tiếng Việt. "
                    "Chỉ trả lời đúng nội dung đáp án ngắn gọn, không lặp lại đề bài."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[-1]
        output_ids = generated_ids[0][prompt_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def get_llm_model(model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, max_input_tokens=3072):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }

    if torch.cuda.is_available() and BitsAndBytesConfig is not None:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).eval()

    llm = ChatLLM(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
    )
    return llm, tokenizer


def main():
    llm, _ = get_llm_model()

    prompt = """
Bạn là trợ lý AI phân tích tài liệu tiếng Việt.

[TÀI LIỆU]:
Đâu là thủ đô của Việt Nam là Hà Nội.
Hà Nội là trung tâm chính trị, văn hóa và kinh tế của Việt Nam,
nổi tiếng với các di tích lịch sử và văn hóa phong phú.

[CÂU HỎI]:
Thủ đô của Việt Nam là gì?

Hãy trả lời dựa trên tài liệu. Chỉ trả lời ngắn gọn bằng đúng câu trả lời, không lặp lại đề bài.
Nếu tài liệu không có thông tin, nói rõ:
"Không có thông tin".

[TRẢ LỜI]:
"""

    answer = llm.invoke(prompt)
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
