import re
from pydantic import BaseModel, EmailStr, validator, Field
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

    @validator("password")
    def validate_password(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least 1 uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least 1 lowercase letter.")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least 1 number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least 1 special character.")
        return v

class UserAuth(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
