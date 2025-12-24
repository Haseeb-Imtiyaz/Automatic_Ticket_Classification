# Automatic Ticket Classification API

FastAPI application for classifying customer complaint tickets into categories.

## Categories

- Bank account services
- Credit card / Prepaid card
- Mortgages/loans
- Theft/Dispute reporting
- Others

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Run the application:
```bash
uvicorn main:app --reload
```

3. Access the API at `http://localhost:8000`

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /topics` - Get all topic categories
- `POST /predict` - Classify a ticket complaint

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "I want to open a joint account for my family"}'
```

## Deployment on Render

1. Push your code to a Git repository
2. Create a new Web Service on Render
3. Connect your repository
4. Render will automatically detect the `render.yaml` configuration
5. The service will be deployed automatically

