from dataclasses import dataclass


@dataclass
class Config:
    x_pad: int
    x_query: int
    x_center: int
    x_max: int

    @classmethod
    def get(cls, is_half: bool) -> "Config":
        return (
            cls(
                x_pad=3,
                x_query=10,
                x_center=60,
                x_max=65,
            )
            if is_half
            else cls(
                x_pad=1,
                x_query=6,
                x_center=38,
                x_max=41,
            )
        )
