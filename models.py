from typing import List
from pydantic import BaseModel, Field


class AddItemForm(BaseModel):
    item_id: str = Field(default=None)
    imageUrls: List[str] = Field(default=None)
