import os
import sys
import traceback
import time
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
Analyze the Python exception and identify the exact line
number(s) in the submitted code that caused the error.

Rules:
1. Count lines in the submitted code starting from line 1.
2. Do not count traceback lines or lines from main.py.
3. Return only line numbers belonging to the submitted code.
4. For a NameError, identify the line using the undefined name.
5. For a SyntaxError, identify the line containing the syntax error.
6. Return an empty list only if no submitted-code line can
   reasonably be identified.
7. Return the result in the required JSON format.

SUBMITTED CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ErrorAnalysis
                )
            )

            result = ErrorAnalysis.model_validate_json(
                response.text
            )

            # Keep only valid submitted-code line numbers
            valid_lines = sorted({
                line for line in result.error_lines
                if isinstance(line, int)
                and 1 <= line <= len(code.splitlines())
            })

            return valid_lines

        except Exception as e:
            if attempt == 2:
                print(
                    f"Gemini analysis failed: "
                    f"{type(e).__name__}: {e}"
                )
                return []

            # Retry temporary errors with a short delay
            time.sleep(attempt + 1)

    return []


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
