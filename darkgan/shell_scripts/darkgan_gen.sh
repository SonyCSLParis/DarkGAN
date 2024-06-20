# random gen
# python generate.py random -d output_networks/darkgan/darkGAN_T1_small --single-file --val -N 30 -o ~/developer/darkgan_audios_web
# python generate.py random -d output_networks/darkgan/darkGAN_T1.5_small --single-file --val -N 30 -o ~/developer/darkgan_audios_web
# python generate.py random -d output_networks/darkgan/darkGAN_T2_small --single-file --val -N 30 -o ~/developer/darkgan_audios_web
# python generate.py random -d output_networks/darkgan/darkGAN_T3_small --single-file --val -N 30 -o ~/developer/darkgan_audios_web
# python generate.py random -d output_networks/darkgan/darkGAN_T5_small --single-file --val -N 30 -o ~/developer/darkgan_audios_web

# # scales
# python generate.py scale -d output_networks/darkgan/darkGAN_T1_small --single-file --val -o ~/developer/darkgan_audios_web
# python generate.py scale -d output_networks/darkgan/darkGAN_T1.5_small --single-file --val -o ~/developer/darkgan_audios_web
# python generate.py scale -d output_networks/darkgan/darkGAN_T2_small --single-file --val -o ~/developer/darkgan_audios_web
# python generate.py scale -d output_networks/darkgan/darkGAN_T3_small --single-file --val -o ~/developer/darkgan_audios_web
# python generate.py scale -d output_networks/darkgan/darkGAN_T5_small --single-file --val -o ~/developer/darkgan_audios_web

# # att sweep
# python -m evaluation.gen_tests.darkgan_att_sweep -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.darkgan_att_sweep -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.darkgan_att_sweep -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.darkgan_att_sweep -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.darkgan_att_sweep -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web

# # max true att
# python -m evaluation.gen_tests.true_max_att -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.true_max_att -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.true_max_att -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.true_max_att -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.true_max_att -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web

# # fixed pf rand z
# python -m evaluation.gen_tests.fix_pf_rand_z -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.fix_pf_rand_z -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.fix_pf_rand_z -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.fix_pf_rand_z -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python -m evaluation.gen_tests.fix_pf_rand_z -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web

# # # timbre transfer
python -m evaluation.gen_tests.timbre_transfer -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web -t ~/developer/darkgan_audios_web/true_ref/
python -m evaluation.gen_tests.timbre_transfer -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web -t ~/developer/darkgan_audios_web/true_ref/
python -m evaluation.gen_tests.timbre_transfer -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web -t ~/developer/darkgan_audios_web/true_ref/
python -m evaluation.gen_tests.timbre_transfer -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web -t ~/developer/darkgan_audios_web/true_ref/
python -m evaluation.gen_tests.timbre_transfer -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web -t ~/developer/darkgan_audios_web/true_ref/


# # # interpolations
# # # attribute interpolation
# python generate.py att_interpolation -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python generate.py att_interpolation -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python generate.py att_interpolation -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python generate.py att_interpolation -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python generate.py att_interpolation -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web 

# # z radial interpolations
# python generate.py radial_interpolation -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python generate.py radial_interpolation -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python generate.py radial_interpolation -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python generate.py radial_interpolation -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python generate.py radial_interpolation -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web 

# z spherical interpolations
# python generate.py spherical_interpolation -d output_networks/darkgan/darkGAN_T1_small -o ~/developer/darkgan_audios_web
# python generate.py spherical_interpolation -d output_networks/darkgan/darkGAN_T1.5_small -o ~/developer/darkgan_audios_web
# python generate.py spherical_interpolation -d output_networks/darkgan/darkGAN_T2_small -o ~/developer/darkgan_audios_web
# python generate.py spherical_interpolation -d output_networks/darkgan/darkGAN_T3_small -o ~/developer/darkgan_audios_web
# python generate.py spherical_interpolation -d output_networks/darkgan/darkGAN_T5_small -o ~/developer/darkgan_audios_web 

