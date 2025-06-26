import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash-8b")

# 評価対象の川柳リスト
senryu_list = [
    "桜咲いて　スマホ片手に　春の便り",
    "新緑の芽　伸びる勢い　未来の色",
    "春の陽気に　洗濯物　乾くのが早い",
    "カエルの合唱　田んぼに響き　春の調べ",
    "猫が日向ぼっこ　春の日は　気持ちいい",
    "花粉症に負けて　春の味方　マスク姿",
    "菜の花畑　黄色に染まり　春の絵",
    "春の嵐で　散る桜も　儚い美",
    "春の風が　吹けば　恋の季節",
    "チューリップ咲いて　街も華やか　春の香り",
]

evaluation_prompt = """
あなたは川柳の専門家として、以下の川柳を評価してください。
各川柳について、面白さを10段階（1〜10）で評価し、その理由を簡潔に説明してください。

評価対象の川柳：
"""

# 各川柳を評価プロンプトに追加
for i, senryu in enumerate(senryu_list, 1):
    evaluation_prompt += f"{i}. {senryu}\n"

evaluation_prompt += """
出力形式：
1. [川柳]
   評価：[1〜10の数値]
   理由：[評価理由の簡潔な説明]

2. [川柳]
   評価：[1〜10の数値]
   理由：[評価理由の簡潔な説明]

（以下10個分続く）

最後に、全体的な評価と総合的なコメントを追加してください。
"""

# APIリクエストの送信
response = model.generate_content(evaluation_prompt)

# 結果の表示
print("川柳の評価結果：")
print(response.text)