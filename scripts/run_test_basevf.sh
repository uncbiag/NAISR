python -m test_toyfunc_vf \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_3' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m test_toyfunc_vf \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge_2' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"


python -m test_toyfunc_vf \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge_vec' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"


python -m test_toyfunc_vf \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment="BaselineVF_1110_torus_3D_mlp_hinge" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="BaselineVF_1110_torus_3D"

python -m test_toyfunc_vf \
    --networksetting='examples/toy/torus/baselinevf.json' \
    --experiment='BaselineVF_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="BaselineVF_1110_torus_3D"

