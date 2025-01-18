import streamlit as st
import openai
import whisper
import spacy
import random
from gtts import gTTS
import tempfile

# Set OpenAI API Key securely
openai.api_key = st.secrets[secrets.OpenAPI]

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Whisper ASR model (for audio transcription)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Function to transcribe past interview recordings
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        transcription = whisper_model.transcribe(temp_audio.name)
        return transcription["text"]

# Function to extract questions from transcripts
def extract_questions(transcript):
    doc = nlp(transcript)
    questions = [sent.text for sent in doc.sents if "?" in sent.text]
    return questions

# Generate AI-based questions using past interviews
def fine_tune_question_generation(question_bank):
    prompt = f"Use the following questions to create diverse and realistic interview questions:\n{question_bank}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].split("\n")

# Streamlit App Layout
st.title("AI Mock Interview App")

# Step 1: Upload Past Interviews for Training
st.header("1. Train the App with Past Interviews")
audio_file = st.file_uploader("Upload a past interview recording (MP3/WAV)", type=["mp3", "wav"])
if audio_file:
    st.write("Transcribing the audio...")
    transcript = transcribe_audio(audio_file)
    st.write("**Transcript:**")
    st.write(transcript)

    st.write("Extracting questions from the transcript...")
    questions = extract_questions(transcript)
    st.write("**Extracted Questions:**")
    for idx, question in enumerate(questions):
        st.write(f"{idx+1}. {question}")

    # Save extracted questions in a question bank
    if "question_bank" not in st.session_state:
        st.session_state["question_bank"] = []
    st.session_state["question_bank"].extend(questions)
    st.write("Questions added to the question bank!")

# Step 2: Generate New Questions Based on Training
st.header("2. Generate New Questions from Training Data")
if "question_bank" in st.session_state:
    question_bank = "\n".join(st.session_state["question_bank"])
    st.write("Generating new questions...")
    new_questions = fine_tune_question_generation(question_bank)
    st.write("**New AI-Generated Questions:**")
    for idx, question in enumerate(new_questions):
        st.write(f"{idx+1}. {question}")
else:
    st.write("No questions in the question bank yet. Upload past interviews to train the app.")

# Step 3: Mock Interview with Scenario-Based Questions
st.header("3. Scenario-Based Mock Interview")
if st.button("Generate a Scenario-Based Question"):
    random_question = random.choice(new_questions if "question_bank" in st.session_state else [
        "Tell me about a challenging project you managed.",
        "How do you handle conflicting stakeholder expectations?"
    ])
    st.write(f"**AI Question:** {random_question}")

# Step 4: Real-Time Feedback
st.header("4. Real-Time Feedback on Responses")
response_text = st.text_area("Type your response here")
if st.button("Analyze Response"):
    feedback_prompt = f"Evaluate this response: '{response_text}' and provide constructive feedback."
    feedback_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": feedback_prompt}]
    )
    feedback = feedback_response["choices"][0]["message"]["content"]
    st.write("**AI Feedback:**", feedback)
