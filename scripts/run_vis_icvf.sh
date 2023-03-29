python -m test_toyfunc_icvf \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m test_toyfunc_icvf \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_mlp_hinge' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"


python -m evaluate \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_mlp_hinge' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m evaluate \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_mlp_hinge' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m visualization \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m visualization \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_mlp_hinge' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="Baseline_1110_torus_3D"

python -m visualization_landmark \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m visualization_landmark \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='BaselineVF_1110_torus_3D_mlp_hinge' \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"

python -m evaluate \
    --networksetting='examples/toy/torus/icvf.json' \
    --experiment='ICVF_1110_torus_3D_siren_hinge' \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="ICVF_1110_torus_3D"



