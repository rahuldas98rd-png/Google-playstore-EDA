from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
import joblib
from pipeline.pipeline import DataPipeline

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reg_model = joblib.load("artifacts/models/rating_model.pkl")
clf_model = joblib.load("artifacts/models/success_model.pkl")

DATA_PATH = "data/raw/googleplaystore.csv"
OUTPUT_PATH = "data/processed"

templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authetication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/train")
async def train_route():
    try:
        pipeline = DataPipeline(DATA_PATH, OUTPUT_PATH)
        pipeline.run()
        return Response("Training is successful.")
    except Exception as e:
        return Response(e)

@app.get("/predict")
def predict(reviews: int, installs: int):
    rating = reg_model.predict([[reviews, installs]])[0]
    success = clf_model.predict([[reviews, installs]])[0]

    return {
        "predicted_rating": float(rating),
        "success": bool(success)
    }