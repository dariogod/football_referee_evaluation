from pydantic import BaseModel
from typing import Literal
from color_conversions import RGBColor255

class BBox(BaseModel):
    x1: int
    y1: int 
    x2: int
    y2: int

class PitchCoord(BaseModel):
    x_bottom_middle: float
    y_bottom_middle: float

class Person(BaseModel):
    id: int
    bbox: BBox
    pitch_coord: PitchCoord | None = None
    gt_role: Literal["referee", "goalkeeper", "player_left", "player_right", "unknown"]

class PersonWithJerseyColor(Person):
    jersey_color: RGBColor255

class PersonWithRole(Person):
    pred_role: dict[str, str] = {
        "rgb": "unknown",
        "lab": "unknown",
        "hsv": "unknown"
    }