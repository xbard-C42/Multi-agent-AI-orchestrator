from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import anthropic
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai.api_key = OPENAI_API_KEY

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo only!
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/council/chat")
async def council_chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    # Responses from each council agent
    gpt_response = await call_gpt4(user_input)
    claude_response = await call_claude(user_input)
    gemini_response = await call_gemini(user_input)

    council_responses = [
        {"agent": "GPT-4", "role": "Proposer", "content": gpt_response},
        {"agent": "Claude 3", "role": "Critic", "content": claude_response},
        {"agent": "Gemini 1.5", "role": "Synthesizer", "content": gemini_response},
    ]

    return {"responses": council_responses}

async def call_gpt4(user_input: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a council member. Build on others' ideas, be constructive, collaborate!"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"[GPT-4 Error: {str(e)}]"

async def call_claude(user_input: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",  # Or "opus", etc.
            max_tokens=256,
            temperature=0.7,
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        return response.content[0].text if response.content else "[Claude returned no content]"
    except Exception as e:
        return f"[Claude Error: {str(e)}]"

async def call_gemini(user_input: str) -> str:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"[Gemini Error: {str(e)}]"
