import os
import sys
import traceback
from io import StringIO
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    output_buffer = StringIO()
    sys.stdout = output_buffer

    try:
        exec(code, {})
        return {
            "success": True,
            "output": output_buffer.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> list[int]:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )

    prompt = f"""
Identify the exact line numbers in the submitted Python
code that caused the exception.

Count submitted code lines starting at 1.
Do not count traceback lines.
Return an empty list only if no code line can be identified.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ErrorAnalysis
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code, execution["output"]
        )
    except Exception as e:
        print(f"Gemini analysis failed: {type(e).__name__}: {e}")
        error_lines = []

    return {
        "error": error_lines,
        "result": execution["output"]
    }
