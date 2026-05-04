"""
Dustbin Flap Control API
Analyzes image + sensor data to determine if waste is wet/dry and flap direction (left/right).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Dustbin Flap Control API",
    description="Analyzes waste images and sensor data to determine flap direction (left for wet, right for dry)",
)


class AnalyzeResponse(BaseModel):
    condition: str = Field(..., description="'wet' or 'dry'")
    flap_direction: str = Field(..., description="'left' or 'right'")
    reasoning: str = Field(..., description="Brief explanation of the decision")
    saved_image_path: str = Field(..., description="Path where the image was saved")


@app.get("/")
async def root():
    return {
        "message": "Dustbin Flap Control API",
        "docs": "/docs",
        "endpoint": "POST /analyze",
    }


async def save_uploaded_image(file: UploadFile) -> tuple[Path, str, bytes]:
    """Save uploaded image to uploads folder. Returns (path, mime_ext, image_bytes)."""
    ext = "jpg"
    if file.content_type and "png" in file.content_type:
        ext = "png"
    elif file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("jpg", "jpeg", "png", "gif", "webp"):
            ext = "jpg"
    ext = "jpg" if ext == "jpeg" else ext

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"waste_{timestamp}.{ext}"
    filepath = UPLOAD_DIR / filename
    image_bytes = await file.read()
    filepath.write_bytes(image_bytes)
    mime_ext = "jpeg" if ext == "jpg" else ext
    return filepath, mime_ext, image_bytes


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(..., description="Image file (jpg, png, etc.)"),
    moisture_data: float = Form(..., description="Soil moisture sensor value (0-4096). Below 3000=wet, 3000+=dry"),
):
    # Save incoming image to uploads folder
    saved_path, mime_ext, image_bytes = await save_uploaded_image(image)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not set. Add it to your .env file.",
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""Analyze this image of soil/waste along with the sensor data below.

SENSOR DATA:
- Soil Moisture (raw 0-4096): {moisture_data}
  Rule: below 3000 = wet, 3000 or above = dry

Determine:
1. Is the material WET or DRY based on the image and sensor data?
2. Should the dustbin flap open LEFT or RIGHT?
   - LEFT = for wet/organic/compostable waste
   - RIGHT = for dry/recyclable waste

Respond with ONLY valid JSON in this exact format (no markdown, no extra text):
{{"condition": "wet" or "dry", "flap_direction": "left" or "right", "reasoning": "brief explanation"}}"""

    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=f"image/{mime_ext}",
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image_part, prompt],
        )

        raw_content = response.text
        if not raw_content:
            raise HTTPException(
                status_code=500,
                detail="Model returned empty response.",
            )
        content = raw_content.strip()
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)
        result["saved_image_path"] = str(saved_path)
        return AnalyzeResponse(**result)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
