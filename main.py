from fastapi import FastAPI
from routers.stock_prediction import router as stock_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stock_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)