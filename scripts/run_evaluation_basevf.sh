

python -m evaluate \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_2' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="BaselineVF_1110_torus_3D"

python -m evaluate \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge_2' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="BaselineVF_1110_torus_3D"





