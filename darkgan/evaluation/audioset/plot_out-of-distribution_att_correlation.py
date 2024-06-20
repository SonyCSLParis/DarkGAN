import click
import ipdb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime


# path_t1 = "output_networks/darkgan/darkGAN_T1_small/dark_knowledge_eval/darkGAN_T1_small_s5_i299994/2021-05-10_00_14/out-of-dist_corr_10:26:59.pkl"
# path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/dark_knowledge_eval/darkGAN_T1.5_small_s5_i199996/2021-05-10_00_14/out-of-dist_corr_10:21:48.pkl"
# path_t2 = "output_networks/darkgan/darkGAN_T2_small/dark_knowledge_eval/darkGAN_T2_small_s5_i299994/2021-05-10_00_15/out-of-dist_corr_10:23:34.pkl"
# path_t3 = "output_networks/darkgan/darkGAN_T3_small/dark_knowledge_eval/darkGAN_T3_small_s5_i299994/2021-05-10_00_25/out-of-dist_corr_10:42:00.pkl"
# path_t5 = "output_networks/darkgan/darkGAN_T5_small/dark_knowledge_eval/darkGAN_T5_small_s5_i299994/2021-05-10_14_23/out-of-dist_corr_20:44:44.pkl"

path_t1 = "output_networks/darkgan/darkGAN_T1_small/dark_knowledge_eval_latest/darkGAN_T1_small_s5_i299994/2021-05-11_03_36/out-of-dist_corr_04:29:01.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/dark_knowledge_eval_latest/darkGAN_T1.5_small_s5_i199996/2021-05-11_03_36/out-of-dist_corr_04:30:41.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/dark_knowledge_eval_latest/darkGAN_T2_small_s5_i299994/2021-05-11_03_50/out-of-dist_corr_04:38:54.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/dark_knowledge_eval_latest/darkGAN_T3_small_s5_i299994/2021-05-11_11_51/out-of-dist_corr_12:20:51.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/dark_knowledge_eval_latest/darkGAN_T5_small_s5_i299994/2021-05-11_03_36/out-of-dist_corr_04:29:55.pkl"


# correlations
path_t1 = "output_networks/darkgan/darkGAN_T1_small/dark_knowledge_eval/darkGAN_T1_small_s5_i299994/2021-05-05_14_35/out-of-dist_corr_21:16:15.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/dark_knowledge_eval/darkGAN_T1.5_small_s5_i199996/2021-05-05_14_35/out-of-dist_corr_01:04:48.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/dark_knowledge_eval/darkGAN_T2_small_s5_i299994/2021-05-05_14_34/out-of-dist_corr_01:01:53.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/dark_knowledge_eval/darkGAN_T3_small_s5_i299994/2021-05-05_14_35/out-of-dist_corr_00:03:53.pkl"


# path_t5 = "output_networks/darkgan/darkGAN_T5_small/dark_knowledge_eval/darkGAN_T5_small_s5_i299994/2021-05-08_18_06/out-of-dist_corr_22:28:33.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/dark_knowledge_eval/darkGAN_T5_small_s5_i299994/2021-05-08_13_05/out-of-dist_corr_13:57:15.pkl"

# increment consistency
# path_t1 = "output_networks/darkgan/darkGAN_T1_small/dark_knowledge_eval_latest2/darkGAN_T1_small_s5_i299994/2021-05-16_05_18/out-of-dist_corr_06:15:02.pkl"
# path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/dark_knowledge_eval_latest2/darkGAN_T1.5_small_s5_i199996/2021-05-16_05_17/out-of-dist_corr_06:13:59.pkl"
# path_t2 = "output_networks/darkgan/darkGAN_T2_small/dark_knowledge_eval_latest2/darkGAN_T2_small_s5_i299994/2021-05-16_05_19/out-of-dist_corr_06:11:04.pkl"
# path_t3 = "output_networks/darkgan/darkGAN_T3_small/dark_knowledge_eval_latest2/darkGAN_T3_small_s5_i299994/2021-05-16_05_19/out-of-dist_corr_06:15:18.pkl"
# path_t5 = "output_networks/darkgan/darkGAN_T5_small/dark_knowledge_eval_latest2/darkGAN_T5_small_s5_i299994/2021-05-16_06_37/out-of-dist_corr_07:06:01.pkl"


@click.command()
@click.option('-d', '--out-dir', type=click.Path(exists=True), required=True, default='')
def main(out_dir):
	table_t1 = pd.read_pickle(path_t1)

	# table_t1['p-t'] = table_t1['p-t'].astype('double')
	# table_t1['real_offset'] = table_t1['real_offset'].astype('double')
	# table_t1['pred_std'] = table_t1['pred_std'].astype('double')
	table_t15 = pd.read_pickle(path_t15)
	# table_t15['p-t'] = table_t15['p-t'].astype('double')
	# table_t15['real_offset'] = table_t15['real_offset'].astype('double')
	# table_t15['pred_std'] = table_t15['pred_std'].astype('double')
	table_t2 = pd.read_pickle(path_t2)
	# table_t2['p-t'] = table_t2['p-t'].astype('double')
	# table_t2['real_offset'] = table_t2['real_offset'].astype('double')
	# table_t2['pred_std'] = table_t2['pred_std'].astype('double')
	table_t3 = pd.read_pickle(path_t3)
	# table_t3['p-t'] = table_t3['p-t'].astype('double')
	# table_t3['real_offset'] = table_t3['real_offset'].astype('double')
	table_t5 = pd.read_pickle(path_t5)
	# table_t5['p-t'] = table_t5['p-t'].astype('double')
	# table_t5['real_offset'] = table_t5['real_offset'].astype('double')
	# table_t5['pred_std'] = table_t5['pred_std'].astype('double')


	corr_t1 = table_t1.groupby('increment').agg('mean')
	corr_t1['temperature'] = 1
	
	corr_t15 = table_t15.groupby('increment').agg('mean')
	corr_t15['temperature'] = 1.5
	
	corr_t2 = table_t2.groupby('increment').agg('mean')
	corr_t2['temperature'] = 2
	
	corr_t3 = table_t3.groupby('increment').agg('mean')
	corr_t3['temperature'] = 3
	
	corr_t5 = table_t5.groupby('increment').agg('mean')
	corr_t5['temperature'] = 5

	corr_all = pd.concat(
		[corr_t1, corr_t15, corr_t2, corr_t3, corr_t5],
		axis=0)
	# corr_all = pd.concat(
	# 	[corr_t1, corr_t15, corr_t2, corr_t5],
	# 	axis=0)
	# ipdb.set_trace()
	plt.figure()
	ax = sns.lineplot(data=corr_all, x="increment", y="corr", hue="temperature", palette=sns.color_palette('colorblind', 5), style='temperature', dashes=False)
	ax.set(xlabel=r'$\Delta$', ylabel=r'$\frac{F_{t}(G(z,p, \alpha + \Delta)) - F_{t}(G(z,p, \alpha))}{std}$')
	ax.set(xscale='log')
	ax.get_figure().savefig(f'{out_dir}/out-of-dist_att_avg_correlation_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.pdf')
	plt.close()


if __name__ == "__main__":
    main()
