from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Correct model name for English to Amharic
model_name = "Helsinki-NLP/opus-mt-en-cus"  # Cus = Cushitic languages (includes Amharic)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    if not text.strip():
        return "⚠️ Please enter some text."

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"⚠️ Translation error: {str(e)}"

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=3, label="Enter English Text"),
    outputs=gr.Textbox(label="Amharic Translation"),
    title="🌍 English to Amharic Translator",
    description="✔ Powered by Helsinki-NLP model.",
    examples=[
        ["Good morning"],
        ["Thank you very much"],
        ["How much does this cost?"]
    ]
)

if __name__ == "__main__":
    demo.launch()
