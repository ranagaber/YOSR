from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


login(token="")

dataset = load_dataset('ranwakhaled/EgyM3AV')
dataset = dataset['test']

model_id = "RanaGaber/gemma-3-12b-Full-IT-EGY-1E"
model_name = model_id.split("/")[-1]

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

processor = AutoProcessor.from_pretrained(model_id)

BATCH_SIZE = 8

PROMPT_TEMPLATE = (
    "Explain this slide in **Egyptian Arabic** using **clear** and **simple** sentences.\n"
    "Cover all visible elements in the slides including text, formulas, images and diagrams.\n"
    "Do NOT add new information, examples, definitions, or assumptions "
    "that are not explicitly shown on the slide."
)

prompt_str = f"""<start_of_image>
User: {PROMPT_TEMPLATE}
Assistant:"""


lecture_to_indices = defaultdict(list)

for idx, row in enumerate(dataset):
    lecture_to_indices[row["lecture"]].append(idx)

records = []

for lec, indices in tqdm(lecture_to_indices.items(), desc="Lectures"):

    batch_images = []
    batch_rows = []

    for idx in indices:

        row = dataset[idx]

        image = row["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        batch_images.append(image)
        batch_rows.append(row)

        if len(batch_images) == BATCH_SIZE:

            inputs = processor(
            images=[[img] for img in batch_images],
            text=[prompt_str] * len(batch_images),
            return_tensors="pt",
            padding=True,
            truncation=False,)

            inputs = {
                k: v.to(model.device)
                for k, v in inputs.items()
            }

            with torch.inference_mode():

                generations = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,

                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,

                    eos_token_id=processor.tokenizer.eos_token_id,)

            input_lengths = (
                inputs["attention_mask"]
                .sum(dim=1)
                .tolist()
            )

            decoded_outputs = []

            for gen, in_len in zip(generations, input_lengths):

                output_tokens = gen[in_len:]

                decoded = processor.decode(
                    output_tokens,
                    skip_special_tokens=True
                )

                decoded_outputs.append(decoded)

            for row_data, decoded in zip(batch_rows, decoded_outputs):

                records.append({
                    "lecture": lec,
                    "image_name": row_data["image_name"],
                    "output": decoded,
                })

            batch_images = []
            batch_rows = []

    if batch_images:

        inputs = processor(
        images=[[img] for img in batch_images],
        text=[prompt_str] * len(batch_images),
        return_tensors="pt",
        padding=True,
        truncation=False,)

        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            generations = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,

                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,

                    eos_token_id=processor.tokenizer.eos_token_id,)

        input_lengths = (
            inputs["attention_mask"]
            .sum(dim=1)
            .tolist()
        )

        for gen, in_len, row_data in zip(
            generations,
            input_lengths,
            batch_rows
        ):

            output_tokens = gen[in_len:]

            decoded = processor.decode(
                output_tokens,
                skip_special_tokens=True
            )

            records.append({
                "lecture": lec,
                "image_name": row_data["image_name"],
                "output": decoded,
            })

df = pd.DataFrame(records)
df.to_csv(f"{model_name}_outputs.csv", index=False, encoding="utf-8-sig")
