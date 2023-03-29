python -m test_toyfunc_lipnaivf \
    --networksetting='examples/toy/torus/lip_naivf.json' \
    --experiment="LipNAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge_1" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="LipNAIVF_TEM_VEC_AnalFit_1110"

python -m test_toyfunc_lipnaivf\
    --networksetting='examples/toy/torus/lip_naivf.json' \
    --experiment="LipNAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_hinge" \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="LipNAIVF_TEM_VEC_AnalFit_1110"


python -m test_toyfunc_lipnaivf \
    --networksetting='examples/toy/torus/lip_naivf.json' \
    --experiment="LipNAIVF_TEM_VEC_AnalFit_1110_torus_3D_mlp_pe_hinge" \
    --posenc \
    --backbone='mlp' \
    --shape='torus' \
    --dim=3 \
    --prefix="LipNAIVF_TEM_VEC_AnalFit_1110"






