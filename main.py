from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import os
from typing import Optional
import re

# Function to download files from a URL
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Attempting to download {filename} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{filename} downloaded successfully!")
        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")
            return False
    else:
        print(f"{filename} already exists, skipping download.")
    return True

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB3x4u31dwLrgTNMLNeb9LWNAJiurinzfA")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "4761e19fa1d5f0e73042f23050507d45")

# Google Drive URL for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1npbtNl_kLQ8Cd2rKcmlcyMKN06IwWzW9"

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    print("Gemini API configured successfully")
except Exception as e:
    print(f"Error configuring Gemini API: {str(e)}")
    raise RuntimeError(f"Failed to configure Gemini API: {str(e)}")

# Load the Random Forest model and label encoders with error handling
try:
    # Download the model file
    if download_file(MODEL_URL, "rf_model.joblib"):
        rf_model = joblib.load("rf_model.joblib")
    else:
        raise FileNotFoundError("Failed to download rf_model.joblib")
    
    label_encoders = joblib.load("label_encoders.joblib")
    print("Random Forest model and label encoders loaded successfully")
except Exception as e:
    print(f"Error loading model or encoders: {str(e)}")
    raise RuntimeError(f"Failed to load model or encoders: {str(e)}")

# Load dataset for fallback data
try:
    df = pd.read_csv("updated_india_agri_data.csv")
    print("Dataset loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading dataset: {str(e)}")
    raise FileNotFoundError("updated_india_agri_data.csv not found")

# Emoji mapping for output
emoji_map = {
    "Next Crop": "🌾",
    "Weather": "☁️",
    "Water Needed": "💧",
    "Fertilizer Needed": "🌿",
    "Protect Crops": "🛡️",
    "Limited Resources": "♻️",
    "Disease Prevention Tip": "🩺",
    "Accuracy": "📊",
    "Suggestion": "💡"
}

# Mapping of states to major cities for weather API
STATE_TO_CITY = {
    "Andaman and Nicobar Islands": "Port Blair",
    "Andhra Pradesh": "Amaravati",
    "Arunachal Pradesh": "Itanagar",
    "Assam": "Dispur",
    "Bihar": "Patna",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Raipur",
    "Dadra and Nagar Haveli": "Silvassa",
    "Daman and Diu": "Daman",
    "Delhi": "New Delhi",
    "Goa": "Panaji",
    "Gujarat": "Gandhinagar",
    "Haryana": "Chandigarh",
    "Himachal Pradesh": "Shimla",
    "Jammu and Kashmir": "Srinagar",
    "Jharkhand": "Ranchi",
    "Karnataka": "Bengaluru",
    "Kerala": "Thiruvananthapuram",
    "Ladakh": "Leh",
    "Madhya Pradesh": "Bhopal",
    "Maharashtra": "Mumbai",
    "Manipur": "Imphal",
    "Meghalaya": "Shillong",
    "Mizoram": "Aizawl",
    "Nagaland": "Kohima",
    "Odisha": "Bhubaneswar",
    "Puducherry": "Puducherry",
    "Punjab": "Chandigarh",
    "Rajasthan": "Jaipur",
    "Sikkim": "Gangtok",
    "Tamil Nadu": "Chennai",
    "Telangana": "Hyderabad",
    "Tripura": "Agartala",
    "Uttar Pradesh": "Lucknow",
    "Uttarakhand": "Dehradun",
    "West Bengal": "Kolkata"
}

# Function to fetch live weather data
def fetch_weather(location: str) -> dict:
    city = STATE_TO_CITY.get(location, location)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Weather data for {city}: {data}")
        return {
            "temperature": data["main"]["temp"],
            "rainfall": data.get("rain", {}).get("1h", 0)
        }
    except Exception as e:
        print(f"Error fetching weather for {city}: {str(e)}")
        return {"temperature": 25, "rainfall": 2}

# Function to predict crop using Random Forest
def predict_crop(user_input: dict) -> tuple[str, float]:
    try:
        rf_input = pd.DataFrame([user_input])
        for col in rf_input.columns:
            if col in label_encoders and rf_input[col].dtype == 'object':
                if user_input[col] in label_encoders[col].classes_:
                    rf_input[col] = label_encoders[col].transform([user_input[col]])[0]
                else:
                    rf_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        prediction = rf_model.predict(rf_input)[0]
        probabilities = rf_model.predict_proba(rf_input)[0]
        max_prob = max(probabilities) * 100
        predicted_crop = label_encoders['Recommend Crop'].inverse_transform([prediction])[0]
        print(f"Predicted crop: {predicted_crop}, Confidence: {max_prob}%")
        return predicted_crop, max_prob
    except Exception as e:
        print(f"Error in predict_crop: {str(e)}")
        raise

# Route to handle favicon request
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# Route to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle form submission and provide advice
@app.post("/get_advice", response_class=HTMLResponse)
async def get_advice(
    request: Request,
    state: str = Form(...),
    soil_type: str = Form(...),
    previous_crop: str = Form(...),
    fertilizer_used: str = Form(...),
    water_hardness: str = Form(...),
    livestock: str = Form(...),
    resources: str = Form(...)
):
    try:
        print(f"Received form data: state={state}, soil_type={soil_type}, previous_crop={previous_crop}, "
              f"fertilizer_used={fertilizer_used}, water_hardness={water_hardness}, livestock={livestock}, resources={resources}")

        # Fetch live weather data
        weather_data = fetch_weather(state)
        temperature = weather_data["temperature"]
        rainfall = weather_data["rainfall"]

        # Prepare user input for Random Forest
        user_input = {
            'State': state,
            'Soil Type': soil_type,
            'Previous Crop': previous_crop,
            'Fertilizer Used': fertilizer_used,
            'Water Hardness': water_hardness,
            'Livestock': livestock,
            'Resources': resources,
            'Temperature': temperature,
            'Rainfall': rainfall
        }

        # Predict crop using Random Forest
        predicted_crop, max_prob = predict_crop(user_input)

        # Prepare prompt for Gemini API
        prompt = (
            f"Given the following agricultural conditions:\n"
            f"State: {user_input['State']}\n"
            f"Soil Type: {user_input['Soil Type']}\n"
            f"Previous Crop: {user_input['Previous Crop']}\n"
            f"Fertilizer Used: {user_input['Fertilizer Used']}\n"
            f"Water Hardness: {user_input['Water Hardness']}\n"
            f"Livestock: {user_input['Livestock']}\n"
            f"Resources: {user_input['Resources']}\n"
            f"Temperature: {temperature}°C\n"
            f"Rainfall: {rainfall} mm\n"
            f"Predicted Crop: {predicted_crop}\n"
            "Provide a detailed recommendation in this format:\n"
            "Recommended Crop: [crop name]\n"
            "Required Water: [amount] liters/day\n"
            "Required Fertilizer: [fertilizer type]\n"
            "Crop Protection Tip: [protection tip]\n"
            "Limited Resources Tip: [resource tip]\n"
            "Disease Prevention Tip: [prevention tip]\n"
            "If the model confidence is below 83%, provide suggestions:\n"
            "Suggestions:\n"
            "- [suggestion with **key points** in bold]\n"
            "- [suggestion with **key points** in bold]"
        )

        # Call Gemini API
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        print(f"Gemini API response: {response_text}")

        # Parse Gemini response
        result = {}
        accuracy = max_prob
        suggestions = []
        parsing_suggestions = False
        for line in response_text.split('\n'):
            if line.strip().startswith("Suggestions:"):
                parsing_suggestions = True
                continue
            if parsing_suggestions:
                if line.strip():
                    cleaned_suggestion = line.strip().lstrip('*-').strip()
                    cleaned_suggestion = re.sub(r'\*\*(.*?)\*\*', r'<span class="bold-green">\1</span>', cleaned_suggestion)
                    suggestions.append(cleaned_suggestion)
                continue
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'Recommended Crop':
                    result['Recommend Crop'] = value.split('(')[0].strip()
                elif key == 'Required Water':
                    result['Required Water'] = value
                elif key == 'Required Fertilizer':
                    result['Required Fertilizer'] = value
                elif key == 'Crop Protection Tip':
                    result['Crop Protection Tip'] = value
                elif key == 'Limited Resources Tip':
                    result['Limited Resources Tip'] = value
                elif key == 'Disease Prevention Tip':
                    result['Disease Prevention Tip'] = value

        # Fill in missing fields from dataset
        if not all(key in result for key in ['Recommend Crop', 'Required Water', 'Required Fertilizer', 'Crop Protection Tip', 'Limited Resources Tip', 'Disease Prevention Tip']):
            match = df[df['Recommend Crop'] == predicted_crop].iloc[0]
            for key in ['Recommend Crop', 'Required Water', 'Required Fertilizer', 'Crop Protection Tip', 'Limited Resources Tip', 'Disease Prevention Tip']:
                if key not in result:
                    result[key] = str(match[key])

        # Prepare recommendation items for the template
        recommendation_items = [
            {"emoji": emoji_map["Next Crop"], "label": "Next Crop", "value": result['Recommend Crop']},
            {"emoji": emoji_map["Weather"], "label": "Current Weather", "value": f"{temperature}°C, {rainfall} mm rainfall"},
            {"emoji": emoji_map["Water Needed"], "label": "Water Needed", "value": result['Required Water']},
            {"emoji": emoji_map["Fertilizer Needed"], "label": "Fertilizer Needed", "value": result['Required Fertilizer']},
            {"emoji": emoji_map["Protect Crops"], "label": "Protect Crops", "value": result['Crop Protection Tip']},
            {"emoji": emoji_map["Limited Resources"], "label": "Limited Resources", "value": result['Limited Resources Tip']},
            {"emoji": emoji_map["Disease Prevention Tip"], "label": "Disease Prevention Tip", "value": result['Disease Prevention Tip']},
        ]

        # Render the result page
        return templates.TemplateResponse("result.html", {
            "request": request,
            "recommendation_items": recommendation_items,
            "suggestions": suggestions if accuracy < 83.0 else [],
            "error": None
        })

    except Exception as e:
        print(f"Error in get_advice: {str(e)}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "recommendation_items": [],
            "suggestions": [],
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "127.0.0.1"
    print(f"Starting server on http://{host}:{port} - Access it at http://localhost:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)