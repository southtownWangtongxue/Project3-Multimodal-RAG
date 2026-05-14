import pathlib

from gme_inference import GmeQwen2VL
import torch
from PIL import Image
if __name__ == '__main__':
    OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output"/ 'images'
    # ---------------------------- 1. 加载模型 ----------------------------
    # 关键：指定你本地模型的 snapshot 目录
    model_path = "D:/Develop Kit/HuggingFace_models/hub/models--iic--gme-Qwen2-VL-2B-Instruct"
    # 如果上面的目录不存在，去掉 /snapshots/... 直接指向 model 根目录尝试
    # 推荐先打开文件夹确认：D:/Develop Kit/HuggingFace_models/hub/models--iic--gme-Qwen2-VL-2B-Instruct
    # 里面应有一个 snapshots/ 文件夹，进入后有一个长哈希名的子目录，复制其完整路径。

    gme = GmeQwen2VL(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------- 2. 原始测试数据 ----------------------------
    t2i_prompt = 'Find an image that matches the given text.'
    texts = [
        "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
        "Alibaba office.",
    ]

    images =[
        Image.open(OUTPUT_DIR / "Tesla_Cybertruck_damaged_window.jpg").convert("RGB"),
        Image.open(OUTPUT_DIR / "TaobaoCity_Alibaba_Xixi_Park.jpg").convert("RGB"),
    ]
    # ---------------------------- 3. 单模态嵌入（无指令） ----------------------------
    # e_text = gme.get_text_embeddings(texts=texts)      # 默认 is_query=True, 使用 default_instruction
    # e_image = gme.get_image_embeddings(images=images, is_query=False)

    # print('单模态嵌入', (e_text @ e_image.T).tolist())
    # 注意：由于默认使用了 default_instruction，数值可能与原始结果略有不同

    # ---------------------------- 4. 带指令的单模态嵌入 ----------------------------
    # e_query = gme.get_text_embeddings(texts=texts, instruction=t2i_prompt, is_query=True)
    # e_corpus = gme.get_image_embeddings(images=images, is_query=False)

    # print('带指令的单模态嵌入', (e_query @ e_corpus.T).tolist())

    # ---------------------------- 5. 融合模态嵌入 ----------------------------
    e_fused = gme.get_fused_embeddings(texts=texts, images=images)
    print('融合模态嵌入', e_fused)
    print('融合模态嵌入', e_fused.shape)
    # print('融合模态嵌入', e_fused.tolist())
    print('融合模态嵌入', (e_fused @ e_fused.T).tolist())