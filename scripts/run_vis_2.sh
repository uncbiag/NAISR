

python -m visualization_landmark \
    --networksetting='examples/toy/torus/naivf_tem_vec.json' \
    --experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_siren_hinge_1" \
    --backbone='siren' \
    --shape='torus' \
    --dim=3 \
    --prefix="NAIVF_TEM_VEC_AnalFit_1110"


python -m evaluate \
--networksetting='examples/toy/torus/naivf_tem_vec.json' \
--experiment="NAIVF_TEM_VEC_AnalFit_1110_torus_3D_siren_hinge_1" \
--backbone='siren' \
--shape='torus' \
--dim=3 \
--prefix="NAIVF_TEM_VEC_AnalFit_1110"