from .evaluate import Weights

hand_tuned = Weights(
    holes=-5,
    roughness=-0.6,
    lines=5,
    relative_height=-0.7,
    absolute_height=-0.8,
    cumulative_height=-0.5,
    well_count=0,
    movements_required=0,
)

line_gen9 = Weights(
    holes=-1.7996452185203657,
    roughness=-0.586362164198158,
    lines=1.543429682851114,
    relative_height=-0.5300073707812323,
    absolute_height=0.14184787008854816,
    cumulative_height=-1.974187965195153,
    well_count=-1.9673957577507517,
    movements_required=0,
)

score = Weights(
    holes=-1,
    roughness=-1,
    lines=1.0,
    relative_height=-1,
    absolute_height=1,
    cumulative_height=-1,
    well_count=-1,
    movements_required=-1,
)

by_mode = {
    "lines": line_gen9,
    "score": score,
}
