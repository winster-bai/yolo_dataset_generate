# yolo_dataset_generate
本仓库用于本地生成yolo训练格式的数据集。


在开始之前，请配置好comfyui环境。

## 前景图背景图生成
在后台comfyui运行的条件下，运行image_genarate.py（注意修改你的openai key）,使用chatgpt对简单对象生成多样化的文生图提示词，然后自动生成前景图片（位于comfyui/output文件夹）

## 前景背景合成
打开dataset_generate.py，修改原始图片位置以及生成数量，运行后自动生成数据集，生成后的图片和数据集位于images和labels文件夹以及用于训练的config.yaml文件。

你也可以使用两个ipynb笔记本来分步执行以上步骤。