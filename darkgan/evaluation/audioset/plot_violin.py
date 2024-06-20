import click
import ipdb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

# 1 increment
path_t1 = "output_networks/darkgan/darkGAN_T1_small/violin_plot/darkGAN_T1_small_s5_i299994/2021-05-11_12_56/pred_feat1.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/violin_plot/darkGAN_T1.5_small_s5_i199996/2021-05-11_12_48/pred_feat1.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/violin_plot/darkGAN_T2_small_s5_i299994/2021-05-11_12_58/pred_feat1.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/violin_plot/darkGAN_T3_small_s5_i299994/2021-05-11_12_58/pred_feat1.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/violin_plot/darkGAN_T5_small_s5_i299994/2021-05-11_12_58/pred_feat1.pkl"

# 2.5 increment
path_t1 = "output_networks/darkgan/darkGAN_T1_small/violin_plot/darkGAN_T1_small_s5_i299994/2021-05-11_13_33/pred_feat1.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/violin_plot/darkGAN_T1.5_small_s5_i199996/2021-05-11_13_34/pred_feat1.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/violin_plot/darkGAN_T2_small_s5_i299994/2021-05-11_13_35/pred_feat1.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/violin_plot/darkGAN_T3_small_s5_i299994/2021-05-11_13_35/pred_feat1.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/violin_plot/darkGAN_T5_small_s5_i299994/2021-05-11_13_39/pred_feat1.pkl"

# 2.5 increment / only corr > 0
path_t1 = "output_networks/darkgan/darkGAN_T1_small/violin_plot/darkGAN_T1_small_s5_i299994/2021-05-11_13_55/pred_feat1.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/violin_plot/darkGAN_T1.5_small_s5_i199996/2021-05-11_13_55/pred_feat1.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/violin_plot/darkGAN_T2_small_s5_i299994/2021-05-11_13_56/pred_feat1.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/violin_plot/darkGAN_T3_small_s5_i299994/2021-05-11_14_00/pred_feat1.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/violin_plot/darkGAN_T5_small_s5_i299994/2021-05-11_13_57/pred_feat1.pkl"

# diff with respect to 0 / all atts / rand sample in range(2, 4) / no avg
path_t1 = "output_networks/darkgan/darkGAN_T1_small/violin_plot/darkGAN_T1_small_s5_i299994/2021-05-11_15_21/pred_feat1.pkl"
path_t15 = "output_networks/darkgan/darkGAN_T1.5_small/violin_plot/darkGAN_T1.5_small_s5_i199996/2021-05-11_15_25/pred_feat1.pkl"
path_t2 = "output_networks/darkgan/darkGAN_T2_small/violin_plot/darkGAN_T2_small_s5_i299994/2021-05-11_15_25/pred_feat1.pkl"
path_t3 = "output_networks/darkgan/darkGAN_T3_small/violin_plot/darkGAN_T3_small_s5_i299994/2021-05-11_15_34/pred_feat1.pkl"
path_t5 = "output_networks/darkgan/darkGAN_T5_small/violin_plot/darkGAN_T5_small_s5_i299994/2021-05-11_15_34/pred_feat1.pkl"


@click.command()
@click.option('-d', '--out-dir', type=click.Path(exists=True), required=True, default='')
def main(out_dir):
	table_t1 = pd.read_pickle(path_t1)
	table_t15 = pd.read_pickle(path_t15)
	table_t2 = pd.read_pickle(path_t2)
	table_t3 = pd.read_pickle(path_t3)
	table_t5 = pd.read_pickle(path_t5)

	table_t1['temperature'] = 1
	table_t15['temperature'] = 1.5
	table_t2['temperature'] = 2
	table_t3['temperature'] = 3
	table_t5['temperature'] = 5

	corr_all = pd.concat(
		[table_t1, table_t15, table_t2, table_t3, table_t5],
		axis=0)
	corr_all['y1'] = corr_all['y1'].astype('double')
	# corr_all['y'] -= corr_all['y'].min()
	plt.figure()

	ax = sns.violinplot(bw=1, cut=0, scale='width', data=corr_all, x="temperature", y="y1")
	ax.set(xlabel=r'$Temperature$', ylabel=r'$Pred$')
	# ax.set(xscale='log', xlabel='feature increment (std)', ylabel='(prediction - orig)/std')
	ax.get_figure().savefig(f'{out_dir}/violin_plot_{datetime.now().strftime("%y_%m_%d_%H:%M:%S")}.pdf')
	plt.close()


if __name__ == "__main__":
    main()
