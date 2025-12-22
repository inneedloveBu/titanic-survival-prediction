import pandas as pd
import gradio as gr
import joblib

# 加载已保存的模型
try:
    final_model = joblib.load('titanic_model.pkl')
    print("模型加载成功！")
except FileNotFoundError:
    print("错误：未找到模型文件 ‘titanic_model.pkl’。请先运行 model_training.py 生成它。")
    exit()
# 在运行此脚本前，在同一个Python环境里先运行你的 model_training.py，这样 final_model 就会存在。

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked, title, family_size, is_alone, has_cabin):
    # 将输入转换为一个DataFrame，列名需与训练时完全一致
    input_dict = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'Title': [title],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'HasCabin': [has_cabin]
    }
    input_df = pd.DataFrame(input_dict)

    # 使用模型进行预测
    prediction = final_model.predict(input_df)[0]
    prediction_proba = final_model.predict_proba(input_df)[0]

    # 格式化输出结果
    survival = "幸存" if prediction == 1 else "遇难"
    proba_percent = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
    result = f"**预测结果：{survival}**\n"
    result += f"模型置信度：{proba_percent:.2f}%"
    return result

# 创建Gradio界面
with gr.Blocks(title="泰坦尼克号幸存预测器") as demo:
    gr.Markdown("# 🚢 泰坦尼克号乘客生存预测")
    gr.Markdown("请填写乘客信息，模型将预测其生存概率。")

    with gr.Row():
        with gr.Column():
            pclass = gr.Dropdown(choices=[1, 2, 3], value=3, label="客舱等级 (Pclass)", info="1=头等舱, 2=二等舱, 3=三等舱")
            sex = gr.Radio(choices=["male", "female"], value="male", label="性别 (Sex)")
            age = gr.Slider(minimum=0, maximum=100, value=25, step=1, label="年龄 (Age)")
            sibsp = gr.Number(value=0, label="同船兄弟姐妹/配偶数量 (SibSp)", precision=0)
            parch = gr.Number(value=0, label="同船父母/子女数量 (Parch)", precision=0)
        with gr.Column():
            fare = gr.Number(value=7.5, label="船票费用 (Fare)")
            embarked = gr.Dropdown(choices=["C", "Q", "S"], value="S", label="登船港口 (Embarked)", info="C=Cherbourg, Q=Queenstown, S=Southampton")
            title = gr.Dropdown(choices=["Mr", "Mrs", "Miss", "Master", "Rare"], value="Mr", label="称谓 (Title)", info="从姓名中提取")
            family_size = gr.Number(value=1, label="家庭总人数 (FamilySize)", precision=0)
            is_alone = gr.Radio(choices=[0, 1], value=0, label="是否独自一人 (IsAlone)", info="0=否, 1=是")
            has_cabin = gr.Radio(choices=[0, 1], value=0, label="是否有客舱记录 (HasCabin)", info="0=无, 1=有")

    predict_btn = gr.Button("开始预测", variant="primary")
    output = gr.Markdown(label="预测结果")

    predict_btn.click(
        fn=predict_survival,
        inputs=[pclass, sex, age, sibsp, parch, fare, embarked, title, family_size, is_alone, has_cabin],
        outputs=output
    )

    gr.Markdown("---")
    gr.Markdown("> 此演示基于Kaggle泰坦尼克数据集训练的机器学习模型。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) # 在本地所有网络接口上启动，可通过浏览器访问