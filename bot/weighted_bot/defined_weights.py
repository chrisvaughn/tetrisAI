from .evaluate import Weights

hand_tuned = Weights(
    holes=-5,
    roughness=-0.6,
    lines=5,
    relative_height=-0.7,
    absolute_height=-0.8,
    cumulative_height=-0.5,
    well_count=0,
)

line_gen9 = Weights(
    holes=-1.7996452185203657,
    roughness=-0.586362164198158,
    lines=1.543429682851114,
    relative_height=-0.5300073707812323,
    absolute_height=0.14184787008854816,
    cumulative_height=-1.974187965195153,
    well_count=-1.9673957577507517,
)

score = Weights(
    holes=-0.9784901004142754,
    roughness=-0.20583058703675264,
    lines=0.07239557195342317,
    relative_height=-0.20078851504525788,
    absolute_height=0.03139923580205717,
    cumulative_height=-0.9004622817496482,
    well_count=-0.37403437412071083,
)

lines_lookahead = Weights(
    holes=-0.3984786015311016,
    depth_weighted_holes=-0.8479660736093779,
    roughness=-1.2126656149621224,
    lines=0.5622492704252798,
    relative_height=0.24421372200072533,
    absolute_height=-1.4895761696035557,
    cumulative_height=-0.9712900036038574,
    well_count=0.8901100779275567,
    well_cells=-1.4229520928723982,
    deep_well_count=-1.7900565269031214,
    total_cells=-1.1580050131963833,
    total_weighted_cells=-0.002689582719308359,
    row_transitions=-0.7009772496505169,
    move_cost=0.1752694877034121,
)

by_mode = {
    "lines": line_gen9,
    "lines_lookahead": lines_lookahead,
    "score": score,
}
