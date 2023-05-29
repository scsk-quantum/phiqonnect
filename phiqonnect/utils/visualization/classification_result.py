from matplotlib.ticker import MultipleLocator
import seaborn as sns
import matplotlib.pyplot as plt

from ..calcuration.pca import *
from ..dataset_utils.load_classification_datasets import *

def plot_kernel_matrix(kernel_matrix, title=None, save_file=None, figsize=(5, 5), **args):
	"""カーネル行列をプロットする
        
	Parameters
	----------
	kernel_matrix: array
		カーネル行列
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	
	Returns
	-------
	None
	"""
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	axes.xaxis.set_major_locator(MultipleLocator(5))
	axes.yaxis.set_major_locator(MultipleLocator(5))
	im = axes.imshow(np.asmatrix(kernel_matrix), interpolation='nearest', origin='upper', cmap='bwr', vmin=0, vmax=1)
	fig.colorbar(im, shrink=0.85)
	if title is not None: axes.set_title(f'kernel: {title}')
	if save_file is not None: plt.savefig(f'{save_file}')
	plt.show()
	
def plot_confusion_matrix(confusion_matrix, title=None, save_file=None, figsize=(5, 4), class_label=None, **args):
	"""混合行列をプロットする
        
	Parameters
	----------
	confusion_matrix: array
		カーネル行列
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	class_labes: array
		クラスラベルを更新する際に使用する
	
	Returns
	-------
	None
	"""
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	if class_label is None:
		sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
	else:
		sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_label, yticklabels=class_label)
	if title is not None: axes.set_title(f'{title}')
	if save_file is not None: plt.savefig(f'{save_file}')
	plt.show()
	
def dicision_boundary(qsvc, training_input, test_input, label_train, label_test, class_label, distance=0.1, plot_type="gouraud", title=None, save_file=None, figsize=(8, 8)):
	"""決定境界をプロットする
	実機で使用すると膨大なタスク数になるため使用は非推奨
        
	Parameters
	----------
	qsvc: QSVC
		学習させたQSVCオブジェクト
	training_input: array
		学習データ
	test_input: array
		テストデータ
	label_train: array
		学習データの正解ラベル
	label_test: array
		テストデータの正解ラベル
	class_label: array
		ラベル
	distance: int
		可視化間隔
	plot_type: string
		可視化のタイプ(gouraud|flat|contour)
	title: string
		タイトル
	save_file: string
		ファイルとして保存する場合はファイルパス
	figsize: (int, int)
		Figureのサイズ
	
	Returns
	-------
	result: dict
		{"x_axis": x_axis, "y_axis": y_axis, "predicted_labels": labels}
	"""
	import matplotlib
	from matplotlib.colors import ListedColormap
	
	color_list = []
#	 for idx, (color, rgb) in enumerate(matplotlib.colors.BASE_COLORS.items()):
#		 color_list.append(color)
	for idx, (color, rgb) in enumerate(matplotlib.colors.TABLEAU_COLORS.items()):
		color_list.append(color)
	cmap = ListedColormap(color_list[:len(class_label)], name="custom")
		
	X, y, pca = pca_data(training_input, test_input, 2)
	
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
	
	grid = np.arange(-1.5, 1.5+distance, distance)
	xx, yy = np.meshgrid(grid, grid)
	
	X_origin = reverse_pca(np.c_[xx.ravel(), yy.ravel()], pca)
	Z = qsvc.predict(X_origin)
	
	if np.var(X[:, 0]) <= np.var(X[:, 1]):
		xx, yy = yy, xx
		
	if plot_type == "gouraud":
		plt.pcolormesh(xx, yy, Z.reshape(xx.shape), alpha=0.5, shading="gouraud", cmap=cmap, snap=False, vmin=0, vmax=len(class_label))
	elif plot_type == "flat":
		plt.pcolormesh(xx, yy, Z.reshape(xx.shape), alpha=0.5, shading="flat", cmap=cmap, snap=False, vmin=0, vmax=len(class_label))
		plt.contourf(xx, yy, Z.reshape(xx.shape))
	elif plot_type == "contour":
		plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cmap, alpha=0.5)
		plt.contour(xx, yy, Z.reshape(xx.shape), levels=np.array(class_label)-0.5, linewidth=1, cmap=cmap)
		
	labels = list(range(0, len(class_label)))
	fig = make_plot(training_input, test_input, label_train, label_test, labels, fig, class_label)
	plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), borderaxespad=0, ncol=2)
	fig.tight_layout()
	
	if title is not None:
		fig.title(title)
	
	if save_file is not None:
		plt.savefig(save_file)
		
	plt.show()
	
	return {"x_axis": xx, "y_axis": yy, "predicted_labels": Z.reshape(xx.shape)}