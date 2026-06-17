from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
from models import Chat
from database import Base, engine, SessionLocal
from models import User
from schemas import UserCreate, UserLogin
from auth import hash_password, verify_password
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
from fastapi import FastAPI, Depends, HTTPException

Base.metadata.create_all(bind=engine)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(
        username=user.username,
        password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()

    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "message": "Login successful",
        "user_id": db_user.id,
        "username": db_user.username
    }


MODEL_PATH = "emotion_bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
emotion_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
def groq_response(user_text: str, emotion: str,history: str):
    prompt = f"""
Do not explain privacy, safety, or confidentiality unless the user explicitly asks.
Do not mention being an AI, trust, or data privacy in emotional conversations.
Do not ask reflective or meta questions after the user sets a boundary (e.g., "it's personal").
When a boundary is stated, acknowledge it once and move on naturally.
You are a helpful, respectful assistant.
Be calm, supportive, and professional.
Do not flirt.
Use simple English.
Conversation so far:
{history}
Current user message:
{user_text}
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

class EmotionRequest(BaseModel):
    text: str
    user_id: int
@app.post("/predict-emotion")
def predict_emotion(req: EmotionRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return {"emotion": labels[pred_id]}
@app.post("/chat")
def chat(req: EmotionRequest):
    
    db = SessionLocal()
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    db.query(Chat)\
      .filter(Chat.user_id == req.user_id)\
      .filter(Chat.created_at < seven_days_ago)\
      .delete()

    db.commit()

    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = emotion_model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()
    emotion = labels[pred_id]

    
    recent_chats = (
        db.query(Chat)
        .filter(Chat.user_id == req.user_id)
        .order_by(Chat.created_at.desc())
        .limit(5)
        .all()
    )

    history = ""
    for chat_item in reversed(recent_chats):
        history += f"User: {chat_item.message}\n"
        history += f"Bot: {chat_item.reply}\n"

   
    reply = groq_response(req.text, emotion, history)

    chat_entry = Chat(
        user_id=req.user_id,
        message=req.text,
        reply=reply
    )

    db.add(chat_entry)
    db.commit()
    db.close()

    return {
        "emotion": emotion,
        "reply": reply
    }
@app.get("/chat-history/{user_id}")
def get_chat_history(user_id: int, db: Session = Depends(get_db)):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)

    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .filter(Chat.created_at >= seven_days_ago)
        .order_by(Chat.created_at.asc())
        .all()
    )

    return [
        {"message": c.message, "reply": c.reply}
        for c in chats
    ]


 