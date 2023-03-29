python -m evaluate \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_eik_1" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"



python -m evaluate \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_eik" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"



python -m test_toyfunc_naivf_tem_vec \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_eik" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"

python -m visualization \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_eik" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"

python -m visualization_landmark \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_eik" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"


