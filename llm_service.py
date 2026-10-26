import os
import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# This reads the .env file and injects it into os.environ
load_dotenv() 

# Now read them using os.getenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize client (Ensure GEMINI_API_KEY is set in environment)
client = genai.Client(api_key=api_key)

# Define your target schema using Pydantic
class DiseaseTreatment(BaseModel):
    symptoms: str
    organic_treatment: str
    chemical_treatment: str
    preventive_measures: list[str]

def generate_treatment_plan(crop: str, disease: str, is_healthy: bool) -> dict:
    """Uses an LLM to generate real-time treatment guidelines for a detected disease."""

    if is_healthy:
        return {
            "symptoms": f"The {crop} leaf shows no visual signs of pathology.",
            "organic_treatment": "None required. Maintain balanced watering and soil nutrient levels.",
            "chemical_treatment": "None required.",
            "preventive_measures": [
                "Inspect foliage weekly",
                "Ensure proper plant spacing for airflow",
                "Maintain consistent drip irrigation",
            ],
        }

    # Prompt forcing structured JSON output
    prompt = f"""
    You are an expert agronomist. A computer vision model identified the following condition:
    Crop: {crop}
    Disease: {disease}

    Provide a concise, practical treatment plan in valid JSON format matching this structure:
    {{
      "symptoms": "1-2 sentences describing classic leaf symptoms",
      "organic_treatment": "Specific bio-fungicide, natural spray, or cultural practice",
      "chemical_treatment": "Active chemical ingredients or commercial sprays recommended",
      "preventive_measures": ["Actionable tip 1", "Actionable tip 2", "Actionable tip 3"]
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",  # Fast and cost-effective for structured tasks
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiseaseTreatment,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating treatment plan: {e}")
        # Fallback if API call fails
        return {
            "symptoms": f"Infection detected on {crop} ({disease}).",
            "organic_treatment": "Apply general copper-based fungicide or neem oil spray.",
            "chemical_treatment": "Consult local agricultural extension for targeted treatment.",
            "preventive_measures": [
                "Isolate affected foliage",
                "Avoid overhead watering",
            ],
        }
