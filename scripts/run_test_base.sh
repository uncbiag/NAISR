python -m test_toyfunc_base\
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_mlp_hinge" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"


python -m test_toyfunc_base \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment='Baseline_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"


python -m test_toyfunc_base \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment='Baseline_1110_torus_3D_siren_mse' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m test_toyfunc_base\
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_mlp_mse" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"




