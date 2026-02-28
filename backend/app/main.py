from fastapi import FastAPI
from . import models, auth, database
from .database import engine

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FastAPI Authentication System")

# Include the authentication router
app.include_router(auth.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Authentication System"}
