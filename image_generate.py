# 自动化生成文生图，comfyui_workflowV0.2

import openai
import json
import re
import time

from foreground import img_generate
from background import background_generate
import os
from PIL import Image
import shutil


# 文生图参数配置
generate_foreground = True # 是否生成前景图
steps = 50
queue = 10 # 单个任务队列长度
seed = 0
generate_count = 10 # 生成的prompt数量，最终图像数量=generate_count*queue*batch_size
input_object = "一个苹果"
width = 512
height = 512
segment_threshold = 0.5 # 分割算法阈值

generate_background = True # 是否生成背景图
background_count = 5 # 背景图数量
background_width = 1024
background_height = 1024


# 图像输出路径，不可自定义修改
folder_path = "/home/dfrobot/stable-diffusion-comfyui/ComfyUI/output"

# ChatGPT配置
openai.api_key = "your-openai-apikey"  # Replace with your OpenAI API key

base_prompt = '''
你是一个提示语生成机器人，你的任务是将用户输入的物体和表述扩写，比如用户输入“橘子”，那么这个橘子可能是一个完整的橘子，一堆橘子，剥好的橘子，一瓣橘子等等，它可能在强光下，弱光下，自然光线或者昏暗环境等等，也可能是刚成熟，完全成熟，快坏了的橘子等等。又比如如果对象是自行车，则有可能涉及到不同的种类或型号。你需要列举出尽可能多的该物体状态以及形态，将这些可能的模样写入possible_combinations词条，并按照possible_combinations的内容生成该物体的generate_count个正向提示语和一个反向提示语。具体要求如下：
1.生成的提示语必须以用户提供的物体作为主体，所占画面需要超过整体画面的1/2，不得描述与主体无关的内容;
2.用户所提供的不同物体可能会存在不同的状态、形态，但是请不要局限于橘子的场景，也不要局限于光照等，发挥你的想象力生成该物体可能真实环境下的样子。
3.输出的json格式文本必须全部为英文
4.提示语需要最终输入到StableDiffusion进行文生图，因此请尽可能丰富你的提示语，以确保最终图像质量，不要用简单的一句话概括，你可以参考下面这个复杂且详细的提示语，但不要完全照搬："A highly realistic, high-definition image of a full, uncut watermelon with a smooth, textured surface and natural colors. The watermelon displays shades of green with darker green stripes, and has a few small natural blemishes and imperfections that add to its authenticity. The shape is round and slightly elongated. The lighting is soft and natural, highlighting the watermelon’s texture and shape without overexposure or harsh shadows. The background is a simple, clean white to emphasize the entire watermelon".
5.正向提示词需要强调真实环境，以确保AI生成的图像接近真实照片。
6.生成的对象图片必须是完整的，禁止出现被截断等现象，请在提示语中强调这一点。
7.反向提示语主要针对画面质量，尽可能避免高饱和度、失真的图像出现，也要避免出现多个物体占满画面的情况发生。
8.绝对不可以颠覆用户原本的描述词，只能在原有的描述上进行想象和扩写，比如用户输入为“腐烂的洋葱”，那你的possible_combinations必须要有“腐烂的”这一特征和“洋葱”这一对象。
9.正确使用描述词，避免后续的文生图模型输出不正确的图像。
10.回复除了json列表外不要生成任何其他内容,所有内容包含在{}内。
11.物体背景简单一些，最好为纯白色的背景。

输出请参考下面的格式：
{
possible_combinations:
positive_prompt1:
positive_prompt2:
positive_prompt3:
...
negative_prompt:
}

现在按照要求请生成有关下面物品的提示语：
input_object

'''

background_prompt = '''
你是一个提示语生成机器人，你的任务是思考用户输入的物体，并给出一些相关的背景画面，比如用户输入“橘子”，你要联想出橘子可能出现的背景画面，比如厨房、野餐垫、菜市场等等，请列举出background_count个可能的背景画面并将其丰富并扩写,然后生成为json字典格式内容。具体要求如下：
1.生成的提示语中不可以包含用户提供的物体！这一点非常重要，背景画面不可以出现用户提供的物体。
2.发挥想象力，尽可能丰富你的提示语，让AI生成的图像更加生动。
3.输出的json格式文本必须全部为英文
4.画面质量要真实，模拟现实环境，尽可能避免高饱和度、失真的图像出现，也要避免出现多个物体占满画面的情况发生。
5.正确使用描述词，避免后续的文生图模型输出不正确的图像。
6.回复除了json列表外不要生成任何其他内容,所有内容包含在{}内。
7.物体背景简单一些，最好为纯白色的背景。

输出请参考下面的格式：
{
background_prompt1:
background_prompt2:
background_prompt3:
...
}

现在按照要求请生成有关下面物品的背景的提示语：
input_object

'''


def llm(message):
    completion = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",  # Replace with the GPT-4 model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}

            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Knock knock."},
            # {"role": "assistant", "content": "Who's there?"},
            # {"role": "user", "content": "Orange."},
        ]
    )
    response = completion['choices'][0]['message']['content']
    return response

def getprompt(response):
    positive_prompt = {}
    negative_prompt = ""
    lines = response.split('\n')
    for line in lines:
        if "positive_prompt" in line:
            key, value = line.split(":")
            positive_prompt[key.strip()] = value.strip()
        if "negative_prompt" in line:
            negative_prompt = line.split(":")[1].strip()
    return positive_prompt, negative_prompt

def blankimg_delate():
    # 删除空图像文件
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Open the image using PIL
            image = Image.open(file_path)
            
            # Check if the image has an alpha channel
            if image.mode == "RGBA":
                # Get the alpha channel as a separate image
                alpha = image.split()[3]
                
                # Calculate the number of non-transparent pixels
                non_transparent_pixels = sum(1 for pixel in alpha.getdata() if pixel > 0)
                
                # Check if the number of non-transparent pixels is less than 100
                if non_transparent_pixels < 5000:
                    # Delete the image file
                    os.remove(file_path)
                    print(f"Deleted {filename}")


# response='''
# json
# {
#   "possible_combinations": "a sleeping cat on a soft cushion, a playful kitten chasing a ball of yarn, a cat sitting by a window bathed in sunlight, a curious cat exploring the garden, a majestic cat perched on a high lookout, a fluffy cat being gently petted, a sleek black cat slinking through shadows, a cat with striking green eyes staring intently, a tabby cat basking in a patch of sunlight, a cat grooming itself meticulously, an elegant cat with a long, flowing tail",
#   "positive_prompt1": "A highly realistic, high-definition image of a sleeping cat curled up on a soft, plush cushion. The cat's fur is smooth and well-groomed, with natural colors ranging from creamy whites to deep browns. It's in a peaceful, serene environment, with gentle, natural lighting accentuating its peaceful expression and the fluffiness of its fur. The background is a simple, clean white to emphasize the cat's tranquility and beauty.",
#   "positive_prompt2": "A vibrant, life-like photograph of a playful kitten in the midst of chasing a brightly colored ball of yarn across a well-lit room. The kitten's movements are captured in stunning clarity, showing its intense focus and playful nature. Its fur is a mix of soft grays and whites, looking almost silky under the natural light. The background is a stark, clean white, ensuring the kitten and its playful antics are the focal point of the image.",
#   "positive_prompt3": "An ultra-high-definition image of a cat sitting gracefully by a sunlit window, its silhouette outlined by the soft, warm light of the morning. The cat's eyes are a striking shade of green, full of depth and curiosity as it gazes outside. Its fur is immaculately groomed, with colors that glow vibrantly under the sunlight. The simple white background serves to highlight the cat's regal posture and serene environment.",
#   "positive_prompt4": "A captivating, true-to-life image of a curious cat exploring a garden, with every blade of grass and petal in vivid detail. The cat moves with elegance, its sleek fur patterned in an array of earth tones that blend seamlessly with the natural surroundings. The lighting is soft and dappled, casting gentle shadows that accentuate the cat's adventurous spirit. A plain white background is used to keep the focus firmly on the cat's exploration.",
#   "positive_prompt5": "A realistic and detailed image of a majestic cat perched on a high lookout, surveying its domain with an air of authority. The cat's fur is thick and luxurious, reflecting a natural color palette that stands out against the simple white background. The lighting is soft yet clear, highlighting the cat's muscular form and the keen focus in its eye. This image captures the essence of feline elegance and strength.",
#   "positive_prompt6": "A high-definition photograph showcasing a fluffy cat receiving gentle pets from an unseen human. The texture of the cat's fur looks incredibly soft and inviting, with each hair glistening slightly under the ambient lighting. The cat's contented expression and half-closed eyes convey a sense of happiness and relaxation. The background is a pure white, ensuring that all attention remains on the tender interaction.",
#   "positive_prompt7": "A photo-realistic image of a sleek black cat moving stealthily through the shadows, its eyes glowing subtly in the dim light. The cat's fur has a healthy sheen, contrasting against the white background and drawing the viewer's eye to its graceful form. The lighting is strategic, enhancing the mysterious aura that surrounds the cat without diminishing the clarity of its features.",
#   "positive_prompt8": "A picture-perfect, ultra-realistic image of a cat with striking green eyes, staring intently at something outside of the frame. The cat’s fur displays an exquisite mix of colors, shining under a natural light source that highlights the intensity of its gaze. The simple, unobtrusive white background ensures that the focus remains squarely on the cat's captivating eyes and beautiful fur.",
#   "positive_prompt9": "An incredibly detailed and realistic image of a tabby cat lounging lazily in a sunny spot, its fur a mixture of warm browns and soft greys. The sun casts gentle highlights across the cat's body, accentuating its relaxed posture and the serene expression on its face. The white background contrasts with the cat's fur and the sunlit area, focusing attention on the contentment evident in the cat's demeanor.",
#   "positive_prompt10": "A meticulously captured high-definition image of a cat grooming itself, its tongue and paws in perfect motion. The clarity of the image reveals the texture of the cat's fur, from the soft undercoat to the sleek outer layers. The natural lighting enhances the cat's dedicated grooming process, with a white background ensuring that the focus remains on this intimate aspect of feline life.",
#   "negative_prompt": "Avoid creating images that saturate the colors of the cat's fur unnaturally, distort its features, or fill the frame with multiple subjects that detract from the focus on the cat. Ensure the cat is depicted in its entirety, without cropping parts of its body, against a simple, white background to prevent distractions."
# }
# '''



base_prompt = base_prompt.replace("input_object", input_object)
base_prompt = base_prompt.replace("generate_count", str(generate_count))
base_prompt = background_prompt.replace("input_object", input_object)
base_prompt = background_prompt.replace("background_count", str(background_count))


# 生成前景图
if generate_foreground:
    print("即将生成前景图，ChatGPT回复中...")
    response = llm(base_prompt)
    print("ChatGPT 回复:", response)
    # Save response to response.json
    with open('/home/dfrobot/stable-diffusion-comfyui/ComfyUI/ComfyUI-to-Python-Extension/response.json', 'w') as f:
        json.dump(response, f)

    # 提取正负提示词并画图
    positive_prompt, negative_prompt = getprompt(response)
    # print("prompt1:", prompt)
    for key in positive_prompt:
        print(positive_prompt[key])
        # def img_generate(positive_prompt,negative_prompt,segment_prompt,width,height,steps,queue,segment_threshold):
        img_generate(positive_prompt[key],negative_prompt,segment_prompt=input_object,width=width,height=height,steps=steps,queue=queue,segment_threshold=segment_threshold)

    blankimg_delate()

    # 前景图移动到指定文件夹
    folder_name = "image/" + input_object
    os.makedirs(folder_name)
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            shutil.move(os.path.join(folder_path, filename), os.path.join(folder_name, filename))

# 生成背景图
if generate_background:
    print("即将生成背景图，ChatGPT回复中...")
    response = llm(background_prompt)
    print("ChatGPT 回复:", response)
    # Save response to response.json
    with open('/home/dfrobot/stable-diffusion-comfyui/ComfyUI/ComfyUI-to-Python-Extension/response.json', 'a') as f:
        json.dump(response, f)
    # 提取背景提示词
    background_prompt = getprompt(response)
    for key in background_prompt:
        print(background_prompt[key])
        background_generate(background_prompt[key],width=background_width,height=background_height)

    # 背景图移动到指定文件夹
    folder_name = "image/background"
    os.makedirs(folder_name)
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            shutil.move(os.path.join(folder_path, filename), os.path.join(folder_name, filename))