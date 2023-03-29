python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_siren_hinge_vec" \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_siren_hinge" \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_mlp_hinge" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_siren_pe_hinge" \
    --posenc \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_mlp_pe_hinge" \
    --posenc \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_siren_mse" \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"


python -m evaluate \
    --networksetting='examples/toy/torus/baseline.json' \
    --experiment="Baseline_1110_torus_3D_mlp_mse" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110"
