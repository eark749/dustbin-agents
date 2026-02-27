# Dustbin Flap Control API

Analyzes waste images and sensor data (moisture, soil) using OpenAI's GPT-4 Vision to determine if material is **wet** or **dry**, and returns whether the dustbin flap should open **left** (wet/organic) or **right** (dry/recyclable).

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key: `OPENAI_API_KEY=sk-...`

3. **Run the server**
   ```bash
   uvicorn main:app --reload
   ```

## API

- **GET /** – Health check
- **POST /analyze** – Analyze image + sensor data

### Request (multipart/form-data)

| Field           | Type   | Description                    |
|----------------|--------|--------------------------------|
| `image`        | File   | Image file (jpg, png, etc.)   |
| `moisture_data`| number | Moisture percentage (0-100)    |
| `temperature`  | number | Soil temperature in Celsius    |
| `humidity`     | number | Soil humidity percentage       |
| `ph`           | number | Soil pH level                  |

### Response

```json
{
  "condition": "wet",
  "flap_direction": "left",
  "reasoning": "Moisture level 42.5% and visual appearance indicate damp organic waste...",
  "saved_image_path": "C:\\...\\uploads\\waste_20250227_143022.jpg"
}
```

## Interactive docs

Open http://127.0.0.1:8000/docs after starting the server.
