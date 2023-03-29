python -m test_toyfunc_naivf_tem_vec \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_siren_hinge" \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"


python -m test_toyfunc_naivf_tem_vec \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment_directory="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_siren_pe_hinge" \
    --posenc \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"

python -m test_toyfunc_naivf_tem_vec\
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"


python -m test_toyfunc_naivf_tem_vec \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_pe_hinge" \
    --posenc \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"




