from pydantic import BaseModel, PositiveFloat

class IrisData(BaseModel):
    sepal_length:PositiveFloat
    sepal_width:PositiveFloat
    petal_length:PositiveFloat
    petal_width:PositiveFloat
    