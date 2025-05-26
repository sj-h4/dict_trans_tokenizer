from pydantic import BaseModel


class BilingualDict(BaseModel):
    entry: str
    definitions: list[str]
