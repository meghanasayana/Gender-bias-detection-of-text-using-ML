import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

print("🌐 Initializing Gender Bias Detection Web App...")

# Load trained model if available
tokenizer = None
model = None

if os.path.exists("model"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./model")
        model = AutoModelForSequenceClassification.from_pretrained("./model")
        model.eval()
        model_status = "✅ Using your trained RoBERTa model"
        print("Model loaded successfully!")
    except:
        model_status = "⚠️ Demo mode - simulated results"
else:
    model_status = "ℹ️ Demo mode - simulated results"

def analyze_gender_bias(text):
    """Main function to analyze text for gender bias"""
    if not text.strip():
        return "Please enter text to analyze.", "", "", "", "", ""
    
    # Gender coding detection
    female_words = ["she", "her", "woman", "girl", "female", "wife", "mother", "daughter"]
    male_words = ["he", "him", "man", "boy", "male", "husband", "father", "son"]
    
    text_lower = text.lower()
    has_female = any(word in text_lower for word in female_words)
    has_male = any(word in text_lower for word in male_words)
    
    if has_female:
        gender_coding = "Female-coded"
    elif has_male:
        gender_coding = "Male-coded"
    else:
        gender_coding = "Gender-neutral"
    
    # Bias pattern detection
    bias_patterns = {
        "emotional": "Emotional stereotyping",
        "bossy": "Leadership bias",
        "aggressive": "Assertiveness bias",
        "dramatic": "Emotional stereotyping",
        "hysterical": "Emotional stereotyping", 
        "naturally better": "Ability stereotyping",
        "born to": "Role stereotyping",
        "designed for": "Purpose stereotyping"
    }
    
    found_bias = []
    for pattern, category in bias_patterns.items():
        if pattern in text_lower:
            found_bias.append(f"'{pattern}' ({category})")
    
    # Model prediction
    hate_speech_prob = 0.0
    if model and tokenizer:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                hate_speech_prob = float(probs)
        except:
            pass
    else:
        # Simulate model behavior for demo
        if found_bias:
            hate_speech_prob = 0.6
        elif any(word in text_lower for word in ["hate", "stupid", "idiot"]):
            hate_speech_prob = 0.8
        else:
            hate_speech_prob = 0.15
    
    # Calculate bias score
    bias_score = 0
    if found_bias:
        bias_score += len(found_bias) * 20
    if hate_speech_prob > 0.5:
        bias_score += 25
    if gender_coding != "Gender-neutral" and found_bias:
        bias_score += 15
    
    bias_score = min(bias_score, 90)
    
    # Determine overall status
    if bias_score >= 60:
        status = "🔴 HIGH BIAS DETECTED"
        recommendations = "Consider rewriting with gender-neutral language. Avoid stereotypical associations."
    elif bias_score >= 30:
        status = "🟡 MODERATE BIAS DETECTED"
        recommendations = "Review for potential stereotypes. Consider more inclusive language."
    else:
        status = "🟢 LOW BIAS - GOOD!"
        recommendations = "Excellent use of inclusive language!"
    
    # Format outputs
    bias_details = "\n".join(found_bias) if found_bias else "None detected"
    
    return (
        status,
        f"{bias_score}/100",
        gender_coding,
        f"{hate_speech_prob:.1%}",
        bias_details,
        recommendations
    )

# Create Gradio interface
with gr.Blocks(title="Gender Bias Detection Tool", theme=gr.themes.Soft()) as app:
    gr.HTML("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2E86AB; margin-bottom: 10px;'>🔍 Gender Bias Detection Tool</h1>
        <p style='color: #666; font-size: 18px;'>Analyze text for gender bias and promote inclusive communication</p>
    </div>
    """)
    
    gr.HTML(f"<p style='text-align: center; color: #888;'>{model_status}</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                lines=6,
                placeholder="Enter your text here to analyze for gender bias...\n\nExamples:\n• Job descriptions\n• Social media posts\n• Articles or reviews\n• Marketing copy",
                label="📝 Text to Analyze",
                show_label=True
            )
            
            analyze_btn = gr.Button("🔍 Analyze for Gender Bias", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    "She is so emotional and can't handle criticism under pressure",
                    "He is naturally better at mathematics and logical reasoning",
                    "This candidate is a skilled software developer with strong leadership qualities",
                    "The team member demonstrates excellent problem-solving abilities and communication skills"
                ],
                inputs=text_input,
                label="💡 Try These Examples"
            )
        
        with gr.Column(scale=1):
            with gr.Group():
                status_output = gr.Textbox(label="📊 Overall Assessment", interactive=False)
                score_output = gr.Textbox(label="📈 Bias Score", interactive=False)
                gender_output = gr.Textbox(label="👤 Gender Coding", interactive=False)
                hate_output = gr.Textbox(label="⚠️ Hate Speech Risk", interactive=False)
                bias_output = gr.Textbox(label="🔍 Specific Bias Indicators", lines=3, interactive=False)
                recommendations_output = gr.Textbox(label="💡 Recommendations", lines=2, interactive=False)
    
    # Event handling
    analyze_btn.click(
        fn=analyze_gender_bias,
        inputs=text_input,
        outputs=[status_output, score_output, gender_output, hate_output, bias_output, recommendations_output]
    )
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; padding: 20px; color: #888; border-top: 1px solid #eee; margin-top: 30px;'>
        <p><strong>About:</strong> This tool uses machine learning to detect potential gender bias in text.</p>
        <p>Promotes awareness for inclusive communication in workplaces, education, and media.</p>
    </div>
    """)

# Launch the application
if __name__ == "__main__":
    print("🚀 Launching Gender Bias Detection Web App...")
    app.launch(share=True, debug=False)
