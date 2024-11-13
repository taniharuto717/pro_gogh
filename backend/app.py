from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import base64
import openai
import os
import re
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
# Memory
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
# チャット履歴保存用のメモリクラスの導入
from langchain.memory import ChatMessageHistory

#flask/langchain
def page_not_found(e):
  return render_template('404.html'), 404
os.environ["OPENAI_API_KEY"] = ""
app = Flask(__name__)
CORS(app)
app.register_error_handler(404, page_not_found)

#langsmith設定
from uuid import uuid4
os.environ["OPENAI_API_KEY"] = "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"
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

#########    対話型鑑賞の処理内容    ######## 
# 共通モデル設定
chat = ChatOpenAI(temperature=0, model_name="gpt-4o")

# ファシリテータ
# prompt
facilitator_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    あなたはファシリテータとして振る舞ってください．私が鑑賞者です．

    鑑賞者に対して一度に2つ以上の質問はしないでください
    対話型鑑賞には入力画像の絵画を用います
    出力に"ファシリテータ:"を含めないでください
    
    対話型鑑賞の流れを以下のような三段階に分けます。
    第一段階：絵画に写っているものから感情を連想し、絵画に対する感情が絵画のどの部分に起因するかを鑑賞者から引き出す
    第二段階：列挙した意見の中から関連のあるものを結びつける
    第三段階：鑑賞者は関連のある意見から新しい解釈を類推する
    
    ファシリテータ:それでは、フィンセント・ファン・ゴッホ作《星月夜》の対話型鑑賞を始めましょう。まず最初に、この絵画を見てどのようなところに注目しましたか？もしくは何を感じましたか？
    {history}
    {input}
    """
)
# memory
facilitator_memory = ConversationBufferWindowMemory(k=3, ai_prefix="ファシリテータ",human_prefix="鑑賞者") #memoryとしてインスタンスを作っている、初期化
facilitator_conversation = ConversationChain(
  llm=chat, 
  memory=facilitator_memory,  
  prompt=facilitator_prompt,
  )

# 対話型鑑賞の段階判定
# プロンプト
PhaseJudge_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    タスク
      art_strに基づいて、ファシリテータが次に振る舞うべき役割を提案する
    
    art_strは対話型鑑賞で鑑賞者が列挙した意見を、絵画要素ごとにまとめたものです
    art_str = {input}
    
    対話型鑑賞でのファシリテータの役割
    1.鑑賞者が絵画にまつわる感想をできるだけ列挙するできるように質問をする
    2.十分に列挙された意見の中から「違う種類の絵画要素」かつ「似たような感想」を持つ意見を関連づけ、鑑賞者に問いかける
    (3つ以上の絵画要素について意見が列挙されていれば「十分」であるとみなす)
    
    出力形式は必ず次に沿ってください。resultには判定結果の整数のみを、reasonには判定した理由を続けてください
    result : 
    reason :
    
    """
)
# PhaseJudgeメモリ
PhaseJudge_memory = ConversationBufferWindowMemory(k=2) #memoryとしてインスタンスを作っている、初期化
PhaseJudge_conversation= ConversationChain(
  llm=chat, 
  memory=PhaseJudge_memory,  
  prompt=PhaseJudge_prompt,
  )

# 鑑賞者の批評状態判定
# prompt
UserCondition_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    以前までの判定結果
    {history}
    タスク
    ・ファシリテータと鑑賞者の対話ログを受けて，鑑賞者の批評状態が1~3のどの段階にあるかを判定してください
    鑑賞者の批評状態を以下のような3つに分ける
        1.反応 (Reaction):鑑賞者が作品に対する直感的・感情的な意見を述べている。あるいは「これを見てどういう感情になった？」に対して主観的な感想を述べている。
        2.知覚的分析 (Perceptual Analysis)は以下の３つのいずれかに該当する場合
            A. 表現 (Representation): 絵画に写っている明示的な物体・事象を記述している。
            B. 形式分析 (Formal Analysis): A.で分析した要素と他の視覚要素との関係性を分析する。そして絵画における最も特徴的な視覚要素を見つける
            C. 形式的特徴付け (Formal Characterization): B.の分析で得られた視覚的要素から、絵画に込められた意図を考察する
        3.個人的解釈 (Personal Interpretation):分析した感情的な意見と視覚的要素を統合して発展的な解釈を述べている
    
    以下にファシリテータと鑑賞者が行っている対話型鑑賞の対話ログを記します．
    {input}
    
    出力形式は必ず次に沿ってください。resultには判定結果の整数のみを、reasonには判定した理由を続けてください
    result : 
    reason : 
    
    """
)
# memory
UserCondition_memory = ConversationBufferWindowMemory(k=5) #memoryとしてインスタンスを作っている、初期化
UserCondition_conversation= ConversationChain(
  llm=chat, 
  memory=UserCondition_memory,  
  prompt=UserCondition_prompt,
  )
# result
# userconditon_result = UserCondition_conversation.predict(input=dialogue_log)
# print(userconditon_result)

# 質問修正(批評状態判定後)
# prompt
Question1_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    タスク
        ・現在の鑑賞者の状態{input}を元に質問を行う
        ・生成はファシリテータの質問内容のみ
        ・対話型鑑賞の流れを以下のような三段階に分けます。
          第一段階：絵画に写っているものから感情を連想し、絵画に対する感情が絵画のどの部分に起因するかを鑑賞者から引き出す
          第二段階：列挙した意見の中から関連のあるものを結びつける
          第三段階：鑑賞者は関連のある意見から新しい解釈を類推する
          今は第一段階にあります
    ルール
        ・鑑賞者の状態が1であれば、現在話題にしている絵画の要素について知覚的分析を行うための質問をしてください。絵画全体に及ぶ質問はしないでください
        ・鑑賞者の状態が2であれば、知覚的分析がより深まるための質問をしてください。
        ・鑑賞者の状態が3であれば、対話の中でまだ話されていない絵画の物体・事象について質問してください
        
    鑑賞者の批評状態
        1.反応 (Reaction):鑑賞者が作品に対する直感的・感情的な意見を述べている。あるいは「これを見てどういう感情になった？」に対して主観的な感想を述べている。
        2.知覚的分析 (Perceptual Analysis)は以下の３つのいずれかに該当する場合
            A. 表現 (Representation): 絵画に写っている明示的な物体・事象を記述している。
            B. 形式分析 (Formal Analysis): A.で分析した要素と他の視覚要素との関係性を分析する。そして絵画の特徴的な視覚要素を見つける
            C. 形式的特徴付け (Formal Characterization): B.の分析で得られた視覚的要素から、絵画に込められた意図を考察する
        3.個人的解釈 (Personal Interpretation):分析した感情的な意見と視覚的要素を統合して発展的な解釈を述べている
        
        対話型鑑賞に使われている絵画：フィンセント・ファン・ゴッホ「星月夜」
    """
)
# メモリはfacilitator_memory
Question1_conversation= ConversationChain(
  llm=chat, 
  memory=facilitator_memory,  
  prompt=Question1_prompt,
  )
 
# 絵画要素
art_dict={
"夜空": "", 
"星": "", 
"月": "", 
"うねる雲": "",
"糸杉の木": "", 
"村の家々": "", 
"教会の塔": "", 
"山": "" 
}

# 絵画要素分類
Classification_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    タスク
      入力する対話型鑑賞のダイアログが，絵画に写っているどの部分に関連があるかを分類してください．
    手順
      ・絵画に写っている要素はart_dictのkeyに書かれています。
      ・鑑賞者の意見が絵画のどの要素についての言及か分類する
      ・鑑賞者の発言内容をvalue、発言内容にまつわる絵画要素をkeyとして出力してください
      ・ファシリテータの発言は分類対象に含めないでください
      ・art_dictにすでに加えている内容は出力しない
        ・art_dictにすでに加えた内容 : {history}
    artdictの初期状態
    art_dict = ｛
    "夜空": "",
    "星": "",
    "月": "",
    "渦巻く雲": "",
    "糸杉の木": "",
    "村の建物": "",
    "教会": "",
    "遠くの山": ""
    ｝
    <対話履歴>{input}
    出力は次の形式に沿って行ってください。
      " "," "
    """)
# Classificationメモリ
Classiification_memory = ConversationBufferWindowMemory(k=2) # memoryとしてインスタンスを作っている、初期化
Classification_conversation= ConversationChain(
  llm=chat, 
  memory=Classiification_memory,  
  prompt=Classification_prompt,
  )

# 関連付けた意見
#プロンプト
associate_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    
    タスク
        ・art_dic内の鑑賞者の意見の中にあるものを関連づける
        ・対話型鑑賞の流れを以下のような三段階に分けます。
          第一段階：絵画に写っているものから感情を連想し、絵画に対する感情が絵画のどの部分に起因するかを鑑賞者から引き出す
          第二段階：列挙した意見の中から関連のあるものを結びつける
          第三段階：鑑賞者は関連のある意見から新しい解釈を類推する
          今は第二段階にあります
    ルール
        「違う種類の絵画要素」かつ「似たような感想」を持つ意見を関連づける
    art_dict = {input}

    出力形式のルール
    ・文字列で出力
    ・resultには関連付けた鑑賞者の意見
    ・関連付ける意見は二つのみ
    ・reasonには判定した理由
    result : 
    reason :
        
    """
)
#メモリ
associate_memory = ConversationBufferWindowMemory(k=2) #memoryとしてインスタンスを作っている、初期化
associate_conversation= ConversationChain(
  llm=chat, 
  memory=associate_memory,  
  prompt=associate_prompt,
  )

# 新しい解釈のための質問
#プロンプト
Question2_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    タスク
        ・入力する二つの感想からどのようなことが考えられるか、を鑑賞者に質問する
        ・二つの感想:{input}
        ・対話型鑑賞の流れを以下のような三段階に分けます。
          第一段階：絵画に写っているものから感情を連想し、絵画に対する感情が絵画のどの部分に起因するかを鑑賞者から引き出す
          第二段階：列挙した意見の中から関連のあるものを結びつける
          第三段階：鑑賞者は関連のある意見から新しい解釈を類推する
        今は第三段階にあります
    ルール
        ・なぜその感想が関連しているのかの理由も示す
        ・ファシリテータは解釈を一切述べない
        ・生成できる質問は一つのみ
        ・生成文に「ファシリテータ」は加えない
        ・文章の生成例
            先ほど発言のあった「A」という感想と「B」という感想は別々の要素に対しての発言ですが、「C」といった部分で共通しているように思えます。この「A」と「B」の関連からどのような解釈が導けそうですか？
        対話型鑑賞に使われている絵画：フィンセント・ファン・ゴッホ「星月夜」
    """
)
#conversation
#メモリはfacilitator_memory
Question2_conversation= ConversationChain(
  llm=chat, 
  memory=facilitator_memory,  
  prompt=Question2_prompt,
  )

# viewer
viewer = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
# プロンプト
viewer_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    役割
      ・対話型鑑賞における鑑賞者
    鑑賞者としてのプロフィール
      ・高校生ぐらいの知能レベル
      ・フレンドリーな発話を行う
    鑑賞絵画
      ・フィンセント・ファン・ゴッホ作《星月夜》
    ルール
      ・対話履歴をもとに鑑賞者として振る舞ってください
      ・ファシリテータの質問に返答する形で感想を述べてください 
      ・「鑑賞者：」は出力には含めないでください
      ・感想は簡潔にまとめてください。出力は100字以内でしてください
    {input}
    """
)
# viewerメモリ
viewer_memory = ConversationBufferMemory(k=5, ai_prefix="鑑賞者", human_prefix="対話履歴") # memoryとしてインスタンスを作っている、初期化
viewer_conversation = ConversationChain(
  llm=viewer, 
  memory=viewer_memory,  
  prompt=viewer_prompt,
  )

# ViewerCorrection
# prompt
viewercorrection_prompt = PromptTemplate(
    input_variables=["history","input"],
    template="""
    {history}
    タスク
        ・鑑賞者の現在の批評状態に則して発言する
        ・生成する内容は鑑賞者の感想のみ
    ルール
        ・ユーザの状態が1であれば、現在話題にしている絵画の要素について直感的な感想を述べてください。感情的かつ直接的な表現を用いてください
        ・ユーザの状態が2であれば、知覚的分析に基づいて、現在話題にしている絵画の要素の特徴を具体的な感想を述べてください
        ・ユーザの状態が3であれば、現在話題にしている絵画の要素の特徴と感情を結びつけて意味を連想させてください
        出力は100字以内にしてください
        
    鑑賞者の批評状態
        1.反応 (Reaction):鑑賞者が作品に対する直感的・感情的な意見を述べている。あるいは「これを見てどういう感情になった？」に対して主観的な感想を述べている。
        2.知覚的分析 (Perceptual Analysis)は以下の３つのいずれかに該当する場合
            A. 表現 (Representation): 絵画に写っている明示的な物体・事象を記述している。
            B. 形式分析 (Formal Analysis): A.で分析した要素と他の視覚要素との関係性を分析する。そして絵画における最も特徴的な視覚要素を見つける
            C. 形式的特徴付け (Formal Characterization): B.の分析で得られた視覚的要素から、絵画に込められた意図を考察する
        3.個人的解釈 (Personal Interpretation):分析した感情的な意見と視覚的要素を統合して発展的な解釈を述べている
        現在の鑑賞者の批評状態：{input}.
        対話型鑑賞に使われている絵画：フィンセント・ファン・ゴッホ「星月夜」
    """
)
viewercorrection_conversation = ConversationChain(
  llm=viewer, 
  memory=viewer_memory,  
  prompt=viewercorrection_prompt,
  )
# art_dict格納
def art_dict_classification(classification_result, art_dict):
    # 行をリストに分割（1行のみの場合でもリストとして扱う）
    lines = classification_result.strip().split("\n")

    # 各行を処理
    for line in lines:
        # 行が空でないことを確認
        if line.strip():
            # カンマで分割
            parts = [item.strip().strip('"') for item in line.split(",")]
            if len(parts) >= 2:  # 最低2つの要素があることを確認
                key = parts[0]
                value = ", ".join(parts[1:])  # 最初の要素以外を結合して値とする
                if key in art_dict:  # keyがart_dictに存在する場合のみ更新
                    art_dict[key] = value

    return art_dict


#########    Flaskの処理内容    ######## 
@app.route('/', methods=['POST'])
def get_response():
    if request.method == 'POST':
        data = request.json
        user = data["text"] # ユーザからの応答
        dialogue_log = facilitator_memory.load_memory_variables({})["history"] # 対話履歴 (["history"]のカラムを呼び出している)
            
        # フェーズの判定
        art_str = json.dumps(art_dict, ensure_ascii=False, indent=4) # 辞書をJSON形式の文字列に変換
        phase  = PhaseJudge_conversation.predict(input=art_str) # フェーズ判定
        phase_result = re.search(r'result\s*:\s*(\d+)', phase).group(1)
        
        if phase_result == "1": # 意見を深堀する段階
          # ファシリテータの発話
          facilitator_conversation.predict(input=user) 
          
          # viewer発話内容
          viewer_conversation.predict(input=dialogue_log)
          viewer_log = viewer_memory.load_memory_variables({})["history"] # 対話履歴
          
          # 鑑賞者の批評状態判定
          dialogue_log = facilitator_memory.load_memory_variables({})["history"] # 対話履歴
          user_condition = UserCondition_conversation.predict(input=dialogue_log)
              
          # 批評状態の段階
          text = user_condition
          usercondition_result = re.search(r'result\s*:\s*(\d+)', text)
          usercondition_result = usercondition_result.group(1) # 1~3の出力のいずれか
          
          # ファシリテータの質問修正
          question_result =  Question1_conversation.predict(input=usercondition_result)
          
          # viewerの感想修正
          viewer_correction = viewercorrection_conversation.predict(input=usercondition_result)
          
          # 絵画要素分類
          classification = Classification_conversation.predict(input=viewer_log) 
          art_dict_classification(classification, art_dict) # 意見を分別
          art_str = json.dumps(art_dict, ensure_ascii=False, indent=4) # 辞書をJSON形式の文字列に変換
          
          # 最終的なファシリテータの返答
          facilitator_res = question_result 
          # 最終的なViewerの返答
          viewer_res = viewer_correction
          
        else: #関連付けた意見を提示
          # 意見を関連づける
          associate_text = associate_conversation.predict(input=art_str)
          associate = re.search(r'result\s*:\s*((?:.|\n)+?)\n\s*reason\s*:\s*((?:.|\n)+)', associate_text)
          associate_result = associate.group(1)
          # 提示
          question2_result = Question2_conversation.predict(input=associate_result)
                                
          # viewer発話内容
          viewer_conve = viewer_conversation.predict(input=dialogue_log)

          # 最終的なファシリテータの返答
          facilitator_res = question2_result 
          # 最終的なViewerの返答
          viewer_res = viewer_conve
            
    return jsonify({"facilitator": facilitator_res, "viewer": viewer_res})

if __name__ == "__main__":
    app.run(debug=True)
