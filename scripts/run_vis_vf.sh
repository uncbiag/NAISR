python -m visualization \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge_2' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m visualization_landmark \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge_2' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m visualization \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_jac' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"


python -m visualization_landmark \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_jac' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"



python -m evaluate \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge_jac' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m evaluate \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_jac' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

