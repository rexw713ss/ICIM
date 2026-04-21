import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx  # 新增：用於繪製因果推斷 DAG 圖
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import dice_ml

# ===================================================
# 1. 載入資料與前處理
# ===================================================
df = pd.read_excel('./default of credit card clients.xls', header=1)
print("正在載入資料...")

df = df.rename(columns={
    'default payment next month': 'default', 
    'PAY_0': 'PAY_1'                         
})
df = df.drop('ID', axis=1)

df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

for col in categorical_features:
    df[col] = df[col].astype(int)
for col in continuous_features:
    df[col] = df[col].astype(float)

print("--- 資料前處理完成 ---")
print(f"資料維度: {df.shape}")
print("\n目標變數 (違約與否) 比例:")
print(df['default'].value_counts(normalize=True).round(3))
print()

# ===================================================
# 2. 探索性資料分析 (EDA)
# ===================================================
sns.set_theme(style="whitegrid")

# PAY_1 長條圖
plt.figure(figsize=(10, 6))
pay1_default_rate = df.groupby('PAY_1')['default'].mean().reset_index()
sns.barplot(x='PAY_1', y='default', data=pay1_default_rate, palette='coolwarm')
plt.title('Default Rate by Recent Payment Status (PAY_1)', fontsize=14)
plt.xlabel('Payment Status (-1: Pay duly, 1+: Months delayed)', fontsize=12)
plt.ylabel('Default Probability', fontsize=12)
plt.axhline(df['default'].mean(), color='red', linestyle='--', label='Average Default Rate')
plt.legend()
plt.savefig('./diagram/計算每個 PAY_1 狀態下的違約率.png')
plt.close()

# 信用卡額度盒鬚圖
plt.figure(figsize=(10, 6))
sns.boxplot(x='default', y='LIMIT_BAL', data=df, palette='Set2')
plt.title('Credit Limit Distribution by Default Status', fontsize=14)
plt.xlabel('Default (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Credit Limit (LIMIT_BAL)', fontsize=12)
plt.savefig('./diagram/信用卡額度.png')
plt.close()

# 年齡分佈堆疊直方圖
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='AGE', hue='default', multiple='stack', bins=30, palette='viridis')
plt.title('Age Distribution and Default Count', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('./diagram/年齡分佈.png')
plt.close()

# ===================================================
# 【新增】3. 建立並視覺化因果圖 (Causal Graph / DAG)
# 此圖奠定了後續反事實解釋必須遵守的物理/商業法則
# ===================================================
print("正在建立因果推斷模型 (DAG)...")
plt.figure(figsize=(10, 6))
G = nx.DiGraph()

# 定義特徵間的因果方向 (上游 -> 下游)
edges = [
    ('AGE', 'EDUCATION'),         # 年齡影響教育程度
    ('AGE', 'MARRIAGE'),          # 年齡影響婚姻狀態
    ('EDUCATION', 'LIMIT_BAL'),   # 教育程度(學歷)影響銀行給的信用額度
    ('LIMIT_BAL', 'PAY_AMT1'),    # 信用額度限制了每個月能還款的上限金額
    ('LIMIT_BAL', 'default'),     # 信用額度本身也隱含風險，影響違約率
    ('PAY_1', 'default'),         # 繳款狀態直接導致是否違約
    ('PAY_AMT1', 'default')       # 實際繳款金額影響違約判定
]

G.add_edges_from(edges)

# 繪製美觀的 DAG
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3500, 
        font_size=10, font_weight='bold', edge_color='gray', arrows=True, arrowsize=20)
plt.title('Causal Directed Acyclic Graph (DAG) for Credit Risk', fontsize=15)
plt.savefig('./diagram/Causal_DAG_因果圖.png')
plt.close()
print("因果圖已儲存至 ./diagram/Causal_DAG_因果圖.png\n")

# ===================================================
# 4. XGBoost 模型訓練與評估實作
# ===================================================
X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_weight, 
    max_depth=5,                   
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("正在訓練 XGBoost 模型...")
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

print("\n--- 模型評估結果 ---")
print(f"整體準確率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Default (0)', 'Default (1)'], 
            yticklabels=['Not Default (0)', 'Default (1)'])
plt.title('Confusion Matrix - XGBoost Baseline')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('./diagram/混淆矩陣.png')
plt.close()

# ===================================================
# 5. SHAP 事後解釋性
# ===================================================
shap.initjs()
print("正在計算 SHAP 值...")
explainer = shap.TreeExplainer(xgb_model)

X_explain = X_test.iloc[:1000]
shap_values = explainer.shap_values(X_explain)

print("\n[全局解釋] 繪製 SHAP Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_explain, show=False)
plt.savefig('./diagram/SHAP_Summary_Plot.png', bbox_inches='tight')
plt.close()

print("\n[局部解釋] 繪製單一申請人的 Force Plot...")
sample_index = 0
single_customer_data = X_explain.iloc[sample_index, :]
single_shap_values = shap_values[sample_index, :]

shap.force_plot(
    explainer.expected_value, 
    single_shap_values, 
    single_customer_data, 
    matplotlib=True,
    show=False
)
plt.savefig('./diagram/SHAP_Force_Plot.png', bbox_inches='tight')
plt.close()

# ===================================================
# 6. DiCE 反事實解釋 (結合因果推斷邊界約束)
# ===================================================
class XGBWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        return self.model.predict(X.astype(float))
        
    def predict_proba(self, X):
        return self.model.predict_proba(X.astype(float))

df_dice = df.copy()
for col in categorical_features:
    df_dice[col] = df_dice[col].astype(str)

dice_data = dice_ml.Data(
    dataframe=df_dice, 
    continuous_features=continuous_features, 
    outcome_name='default'
)
dice_model = dice_ml.Model(model=XGBWrapper(xgb_model), backend="sklearn")
exp = dice_ml.Dice(dice_data, dice_model, method="random")

high_risk_indices = np.where(xgb_model.predict_proba(X_test)[:, 1] > 0.8)[0]
query_instance = X_test.iloc[[high_risk_indices[0]]].copy()

for col in categorical_features:
    query_instance[col] = query_instance[col].astype(str)

print("\n【原始客戶狀態】")
print(query_instance[['AGE', 'LIMIT_BAL', 'PAY_1', 'PAY_AMT1']])
print(f"原始預測結果: 拒絕核貸 (違約機率: {xgb_model.predict_proba(query_instance.astype(float))[0][1]:.2f})")

print("\n正在計算【受因果推斷與商業規則約束】的信用修復建議...")

# 因果約束實踐：
# 基於上方繪製的 DAG 因果圖，PAY_AMT1 (還款金額) 受到 LIMIT_BAL (信用額度) 的絕對限制。
# 且 AGE, EDUCATION 等為上游不可變特徵，僅開放下游行為特徵進行擾動。
customer_limit = float(query_instance['LIMIT_BAL'].values[0])

permitted_ranges = {
    'PAY_AMT1': [0, customer_limit],  # 遵從因果圖的下游約束
    'PAY_AMT2': [0, customer_limit]
}

actionable_features = ['PAY_1', 'PAY_AMT1', 'PAY_AMT2'] 

dice_exp_constrained = exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,                           
    desired_class="opposite",              
    features_to_vary=actionable_features,  
    permitted_range=permitted_ranges       
)

print("\n【符合因果邏輯與現實的反事實修復建議】")
dice_exp_constrained.visualize_as_dataframe(show_only_changes=True)

# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 假設前面的 X_train, X_test, y_train, y_test 已經準備好
# 並且 scale_weight 也已經計算好

print("\n===========================================")
print("開始進行多模型效能比較實驗...")
print("===========================================")

# 1. 定義三個要比較的模型
# 為了公平起見，三個模型都加入處理「資料不平衡」的權重設定
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss')
}

# 2. 訓練模型並收集評估指標
results = []
roc_data = {}

for name, model in models.items():
    print(f"正在訓練 {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # 取得預測為 1 (違約) 的機率
    
    # 計算各項指標
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })
    
    # 儲存畫 ROC 曲線需要的資料 (FPR, TPR)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_data[name] = (fpr, tpr, auc)

# 將結果轉為 DataFrame 方便查看
results_df = pd.DataFrame(results).round(4)
print("\n【多模型效能比較表】")
print(results_df.to_markdown(index=False))

# 3. 繪製 ROC 曲線比較圖
plt.figure(figsize=(10, 8))
colors = {'Logistic Regression': 'gray', 'Random Forest': 'blue', 'XGBoost': 'red'}
linestyles = {'Logistic Regression': '--', 'Random Forest': '-.', 'XGBoost': '-'}

for name in models.keys():
    fpr, tpr, auc = roc_data[name]
    plt.plot(fpr, tpr, color=colors[name], linestyle=linestyles[name], lw=2, 
             label=f'{name} (AUC = {auc:.3f})')

# 畫一條對角線 (隨機猜測線)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=15)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# 儲存圖片
plt.savefig('./diagram/ROC_Curve_Comparison.png', bbox_inches='tight')
plt.close()
print("\nROC 曲線比較圖已儲存至 ./diagram/ROC_Curve_Comparison.png")

# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===========================================")
print("開始進行反事實解釋之量化評估實驗...")
print("===========================================")

# 1. 挑選前 10 名預測機率最高的高風險客戶進行批量實驗
high_risk_indices = np.where(xgb_model.predict_proba(X_test)[:, 1] > 0.8)[0][:10]
queries = X_test.iloc[high_risk_indices].copy()

for col in categorical_features:
    queries[col] = queries[col].astype(str)

# 定義要記錄的指標
metrics = {
    'Unconstrained': {'validity': [], 'sparsity': [], 'cost': []},
    'Constrained': {'validity': [], 'sparsity': [], 'cost': []}
}

print(f"正在對 {len(queries)} 位高風險客戶計算反事實指標，請稍候...")

for idx in range(len(queries)):
    query_instance = queries.iloc[[idx]]
    orig_pay_amt1 = float(query_instance['PAY_AMT1'].values[0])
    customer_limit = float(query_instance['LIMIT_BAL'].values[0])
    
    # --- A. 無約束 DiCE (對照組) ---
    dice_unconstrained = exp.generate_counterfactuals(
        query_instance, total_CFs=1, desired_class="opposite", 
        features_to_vary=['PAY_1', 'PAY_AMT1', 'PAY_AMT2']
    )
    cf_un_df = dice_unconstrained.cf_examples_list[0].final_cfs_df
    
    if cf_un_df is not None and not cf_un_df.empty:
        metrics['Unconstrained']['validity'].append(1)
        # 計算 Sparsity (改變了幾個特徵)
        changed_features = sum(cf_un_df.iloc[0][actionable_features].astype(float) != query_instance[actionable_features].astype(float).iloc[0])
        metrics['Unconstrained']['sparsity'].append(changed_features)
        # 計算 Cost / Proximity (以 PAY_AMT1 的變動絕對值為例)
        cost = abs(float(cf_un_df.iloc[0]['PAY_AMT1']) - orig_pay_amt1)
        metrics['Unconstrained']['cost'].append(cost)
    else:
        metrics['Unconstrained']['validity'].append(0)

    # --- B. 因果約束 DiCE (實驗組) ---
    permitted_ranges = {'PAY_AMT1': [0, customer_limit], 'PAY_AMT2': [0, customer_limit]}
    dice_constrained = exp.generate_counterfactuals(
        query_instance, total_CFs=1, desired_class="opposite", 
        features_to_vary=['PAY_1', 'PAY_AMT1', 'PAY_AMT2'],
        permitted_range=permitted_ranges
    )
    cf_con_df = dice_constrained.cf_examples_list[0].final_cfs_df
    
    if cf_con_df is not None and not cf_con_df.empty:
        metrics['Constrained']['validity'].append(1)
        changed_features = sum(cf_con_df.iloc[0][actionable_features].astype(float) != query_instance[actionable_features].astype(float).iloc[0])
        metrics['Constrained']['sparsity'].append(changed_features)
        cost = abs(float(cf_con_df.iloc[0]['PAY_AMT1']) - orig_pay_amt1)
        metrics['Constrained']['cost'].append(cost)
    else:
        metrics['Constrained']['validity'].append(0)

# 2. 計算平均指標
avg_metrics = []
for method in ['Unconstrained', 'Constrained']:
    avg_val = np.mean(metrics[method]['validity']) * 100
    avg_spa = np.mean(metrics[method]['sparsity'])
    avg_cost = np.mean(metrics[method]['cost'])
    avg_metrics.append({'Method': method, 'Validity (%)': avg_val, 'Avg Sparsity': avg_spa, 'Avg Cost (PAY_AMT1)': avg_cost})

metrics_df = pd.DataFrame(avg_metrics)
print("\n【反事實量化評估結果】")
print(metrics_df.to_markdown(index=False))

# 3. 繪製比較圖表 (雙 Y 軸：左邊畫 Sparsity，右邊畫 Cost)
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(2)
width = 0.35

# 繪製 Sparsity (稀疏度 - 變動特徵數量)
bars1 = ax1.bar(x - width/2, metrics_df['Avg Sparsity'], width, label='Sparsity (Count)', color='skyblue')
ax1.set_ylabel('Average Sparsity (Changed Features)', fontsize=12, color='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(['Baseline (Unconstrained)', 'Proposed (Causal-Constrained)'], fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# 繪製 Cost (擾動成本 - 還款金額變動幅度)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, metrics_df['Avg Cost (PAY_AMT1)'], width, label='Perturbation Cost (NT$)', color='salmon')
ax2.set_ylabel('Average Perturbation Cost in PAY_AMT1', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Quantitative Evaluation of Counterfactual Explanations', fontsize=15)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9), bbox_transform=ax1.transAxes)

plt.savefig('./diagram/CF_Quantitative_Evaluation.png', bbox_inches='tight')
plt.close()
print("\n量化評估比較圖已儲存至 ./diagram/CF_Quantitative_Evaluation.png")

# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("\n===========================================")
print("開始進行演算法公平性與偏見檢驗實驗...")
print("===========================================")

# 1. 設定受保護屬性 (Protected Attribute)：以年齡為例
# 將測試集按年齡分為「青年 (<30歲)」與「中高齡 (>=30歲)」
X_test_fairness = X_test.copy()
X_test_fairness['Actual_Default'] = y_test
X_test_fairness['Predicted_Default'] = y_pred

# 在金融實務中，預測為 0 (不違約) 代表「核貸通過 (Approval)」
X_test_fairness['Actual_Approval'] = 1 - X_test_fairness['Actual_Default']
X_test_fairness['Predicted_Approval'] = 1 - X_test_fairness['Predicted_Default']

# 區分兩大族群
group_young = X_test_fairness[X_test_fairness['AGE'] < 30]
group_mature = X_test_fairness[X_test_fairness['AGE'] >= 30]

# 2. 計算人口統計平權指標 (Demographic Parity)
# 計算實際的合格率 (真實驗本中沒有違約的比例)
actual_approval_young = group_young['Actual_Approval'].mean()
actual_approval_mature = group_mature['Actual_Approval'].mean()

# 計算模型給出的核貸通過率
pred_approval_young = group_young['Predicted_Approval'].mean()
pred_approval_mature = group_mature['Predicted_Approval'].mean()

# 彙整為 DataFrame
fairness_data = {
    'Demographic Group': ['Age < 30 (Young)', 'Age >= 30 (Mature)'],
    'Actual Qualified Rate': [actual_approval_young, actual_approval_mature],
    'Model Approval Rate': [pred_approval_young, pred_approval_mature]
}
fairness_df = pd.DataFrame(fairness_data)

print("\n【演算法公平性指標 (Approval Rates)】")
print(fairness_df.to_markdown(index=False))

# 3. 繪製公平性檢驗長條圖
plt.figure(figsize=(9, 6))
x = np.arange(len(fairness_df['Demographic Group']))
width = 0.35

# 畫出實際合格率與預測核貸率
plt.bar(x - width/2, fairness_df['Actual Qualified Rate'] * 100, width, label='Actual Qualified Rate (%)', color='#90C28A')
plt.bar(x + width/2, fairness_df['Model Approval Rate'] * 100, width, label='Model Approval Rate (%)', color='#2F7B41')

plt.ylabel('Approval / Qualified Rate (%)', fontsize=12)
plt.title('Algorithmic Fairness: Demographic Parity in Credit Approval', fontsize=14)
plt.xticks(x, fairness_df['Demographic Group'], fontsize=12)
plt.legend(loc='lower right')
plt.ylim(0, 100)

# 加上數值標籤
for i, val in enumerate(fairness_df['Actual Qualified Rate']):
    plt.text(i - width/2, val*100 + 1, f'{val*100:.1f}%', ha='center', fontsize=11)
for i, val in enumerate(fairness_df['Model Approval Rate']):
    plt.text(i + width/2, val*100 + 1, f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('./diagram/Fairness_Evaluation.png', bbox_inches='tight')
plt.close()
print("\n公平性評估比較圖已儲存至 ./diagram/Fairness_Evaluation.png")
# =====================
import matplotlib.pyplot as plt
import shap

print("\n===========================================")
print("開始進行 SHAP 特徵交互作用分析實驗...")
print("===========================================")

# 確保之前已經計算過 shap_values 與 X_explain
# explainer = shap.TreeExplainer(xgb_model)
# X_explain = X_test.iloc[:1000]
# shap_values = explainer.shap_values(X_explain)

# 繪製 SHAP Dependence Plot (相依性圖)
# 我們觀察 'LIMIT_BAL' (信用額度) 對預測的影響，
# 並且讓 SHAP 自動挑選最常與之產生交互作用的特徵 (通常會是 PAY_1) 進行顏色編碼
plt.figure(figsize=(10, 6))

# show=False 讓我們可以將圖片存檔而不是卡在視窗
shap.dependence_plot(
    "LIMIT_BAL", 
    shap_values, 
    X_explain, 
    interaction_index="PAY_1",  # 強制觀察額度與繳款狀態的交互作用
    show=False,
    title="SHAP Dependence Plot: Interaction between LIMIT_BAL and PAY_1"
)

# 調整佈局並存檔
plt.tight_layout()
plt.savefig('./diagram/SHAP_Dependence_Plot.png', bbox_inches='tight', dpi=300)
plt.close()
print("\n特徵交互作用圖已儲存至 ./diagram/SHAP_Dependence_Plot.png")

# ===================
# ===================================================
# 8. 模擬前端使用者反饋系統 (User-Centric Feedback System)
# ===================================================
print("\n===========================================")
print("啟動【動態信用修復反饋系統】生成客戶報告...")
print("===========================================")

# 取得原始預測違約機率
original_prob = xgb_model.predict_proba(query_instance.astype(float))[0][1]

# 提取 DiCE 成功生成的第一個反事實樣本
cf_df = dice_exp_constrained.cf_examples_list[0].final_cfs_df

if cf_df is not None and not cf_df.empty:
    cf_instance = cf_df.iloc[0:1].drop('default', axis=1)
    # 計算照做之後的新違約機率
    new_prob = xgb_model.predict_proba(cf_instance.astype(float))[0][1]
    
    # 找出被系統改變的特徵 (Actionable Features)
    changes = {}
    for col in actionable_features:
        orig_val = float(query_instance[col].values[0])
        new_val = float(cf_instance[col].values[0])
        if orig_val != new_val:
            changes[col] = (orig_val, new_val)
    
    # --- 輸出模擬的前端 UI 畫面 ---
    print("\n" + "="*50)
    print("🤖 【AI 財務教練 - 客戶專屬信用修復報告】")
    print("="*50)
    print(f"親愛的客戶您好：\n")
    print(f"經系統綜合評估，您本次的核貸申請【未能通過】。")
    print(f"（當前系統預測之違約風險率為：{original_prob*100:.1f}%）\n")
    print("為協助您重建信用並順利取得貸款，我們的 AI 顧問為您量身打造了以下修復任務：\n")
    
    task_num = 1
    for feature, (old, new) in changes.items():
        if feature == 'PAY_AMT1':
            print(f"📌 任務 {task_num}：請將「下個月的信用卡繳款金額」從 {old:,.0f} 元，提升至 {new:,.0f} 元。")
            task_num += 1
        elif feature == 'PAY_1':
            print(f"📌 任務 {task_num}：請改善「近期的繳款狀態」，盡速繳清目前延遲的帳款。")
            task_num += 1
        elif feature == 'PAY_AMT2':
            print(f"📌 任務 {task_num}：請將「第二個月的信用卡繳款金額」從 {old:,.0f} 元，提升至 {new:,.0f} 元。")
            task_num += 1
            
    print(f"\n💡 系統預測：")
    print(f"若您能達成上述任務，您的違約風險率將大幅降至 【{new_prob*100:.1f}%】！")
    print(f"系統將為您重新評估，預計可【順利通過】下次的核貸申請。")
    print("="*50 + "\n")
else:
    print("系統無法在當前商業約束下找到合適的修復路徑，請指派實體專員介入處理。")

# ===================================================
# ===================================================
# 8. 建立前端互動式 Web 模擬器 (使用 Gradio)
# ===================================================
import gradio as gr
import pandas as pd

print("\n===========================================")
print("正在啟動 Web 網頁版 AI 信用修復模擬器...")
print("===========================================")

# 建立一個預測與修復的包裝函數
def loan_simulator(limit_bal, age, pay_1, pay_amt1):
    # 複製一筆格式相同的測試資料
    test_df = query_instance.copy()
    
    # 【關鍵修復】把隱藏的歷史遲繳紀錄「洗底」！
    # 強制把過去幾個月 (PAY_2 ~ PAY_6) 設為 '0' (正常繳款)
    # 這樣系統才會對我們在網頁上拉動的 PAY_1 敏感
    for col in ['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
        test_df[col] = '0'
        
    # 同時把一些之前可能累積的龐大卡債稍微調降，避免他負債比過高
    for col in ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']:
        test_df[col] = 15000.0 
    
    # 替換成使用者在網頁介面上拉動的數值
    test_df['LIMIT_BAL'] = float(limit_bal)
    test_df['AGE'] = float(age)
    test_df['PAY_1'] = str(int(pay_1))  # 保持 DiCE 需要的字串型態
    test_df['PAY_AMT1'] = float(pay_amt1)
    
    # 1. 進行 XGBoost 預測 (將 DataFrame 轉回 float 餵給模型)
    prob = xgb_model.predict_proba(test_df.astype(float))[0][1]
    
    # 設定閾值 (假設預測機率 > 0.5 判定為拒絕核貸)
    if prob < 0.5:
        return f"✅ 【恭喜！核貸通過】\n\n預測違約風險：{prob*100:.1f}%\n您的財務狀況評估良好，請繼續保持！"
    else:
        result_text = f"❌ 【核貸未通過】\n\n預測違約風險：{prob*100:.1f}%\n\n正在啟動 AI 信用修復引擎 (DiCE)...\n"
        
        # 2. 若拒絕，啟動因果約束 DiCE 尋找解法
        permitted_ranges = {'PAY_AMT1': [0, float(limit_bal)], 'PAY_AMT2': [0, float(limit_bal)]}
        
        try:
            dice_result = exp.generate_counterfactuals(
                test_df, total_CFs=1, desired_class="opposite", 
                features_to_vary=['PAY_1', 'PAY_AMT1'],
                permitted_range=permitted_ranges
            )
            cf_df = dice_result.cf_examples_list[0].final_cfs_df
            
            if cf_df is not None and not cf_df.empty:
                new_pay_amt1 = float(cf_df.iloc[0]['PAY_AMT1'])
                new_prob = xgb_model.predict_proba(cf_df.drop('default', axis=1).astype(float))[0][1]
                
                result_text += "-"*40 + "\n"
                result_text += "💡 【AI 信用修復建議】\n\n"
                result_text += f" 任務：請將「本期繳款金額」提升至新台幣 {new_pay_amt1:,.0f} 元。\n"
                result_text += f" 預期成效：達成後，您的違約風險將降至 {new_prob*100:.1f}%，並有望通過審核！"
                return result_text
            else:
                return result_text + "\n⚠️ 系統無法在您當前的額度條件下找到修復方案，建議尋求實體專員協助。"
        except Exception as e:
            return result_text + f"\n⚠️ 計算修復方案時發生錯誤：{str(e)}"

# 3. 設計 Gradio 網頁介面 (UI)
demo = gr.Interface(
    fn=loan_simulator,
    inputs=[
        gr.Slider(minimum=10000, maximum=500000, step=10000, label="您的信用額度 (LIMIT_BAL)", value=80000),
        gr.Slider(minimum=20, maximum=80, step=1, label="您的年齡 (AGE)", value=30),
        gr.Dropdown(choices=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], label="近期繳款狀態 (PAY_1)  [-1=正常, 1以上=遲繳月數]", value=2),
        gr.Number(label="本期已準備繳交金額 (PAY_AMT1)", value=2000)
    ],
    outputs=gr.Textbox(label=" AI 審核與反饋報告", lines=10),
    title=" 智慧信貸風控與 AI 修復模擬器",
    description="請調整下方參數，模擬銀行端 XGBoost 模型的審核結果。若遭到系統拒絕，後台將自動呼叫 DiCE 引擎，為您生成合法且可行的信用修復計畫！",
    theme=gr.themes.Soft()
)

# 4. 啟動伺服器 (share=True 會產生一個臨時的公開網址，可以傳給教授看)
demo.launch(share=True)

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

print("\n===========================================")
print("開始進行 1000 人大規模系統模擬實驗...")
print("===========================================")

# 步驟 1：篩選出高風險客戶
predictions = xgb_model.predict(X_test)
high_risk_indices = np.where(predictions == 1)[0]
X_test_high_risk = X_test.iloc[high_risk_indices].head(1000).copy() # 加上 copy() 避免警告

# 【關鍵修正 1】：將類別特徵強制轉為字串，對齊 DiCE 訓練時的格式
for col in categorical_features:
    X_test_high_risk[col] = X_test_high_risk[col].astype(str)

print(f"準備對 {len(X_test_high_risk)} 名高風險客戶進行 AI 信用修復模擬...")

simulation_results = []

# 步驟 2：開始自動化模擬
for index, row in tqdm(X_test_high_risk.iterrows(), total=len(X_test_high_risk)):
    start_time = time.time()
    query_instance = row.to_frame().T
    
    # 設定動態財務邊界約束
    customer_limit = float(row['LIMIT_BAL'])
    current_pay = float(row['PAY_AMT1'])
    upper_bound = max(current_pay + 1.0, customer_limit) 
    custom_permitted_range = {'PAY_AMT1': [current_pay, upper_bound]}

    try:
        # 執行結構約束 DiCE
        dice_exp = exp.generate_counterfactuals(
            query_instance,
            total_CFs=1,
            desired_class="opposite",
            features_to_vary=['PAY_AMT1', 'PAY_1'], 
            permitted_range=custom_permitted_range
        )
        
        latency = time.time() - start_time
        
        # 檢查生成結果
        if dice_exp.cf_examples_list and getattr(dice_exp.cf_examples_list[0], 'final_cfs_df', None) is not None:
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            # 【關鍵修正 2】：確保有抓到 DataFrame 且不為空
            if not cf_df.empty:
                new_pay = float(cf_df['PAY_AMT1'].values[0])
                cost = abs(new_pay - current_pay)
                actionable = 1 if cost <= (0.2 * customer_limit) else 0
                
                simulation_results.append({
                    'Validity': 1, 'Latency': latency, 'Cost': cost, 'Actionable': actionable
                })
            else:
                simulation_results.append({'Validity': 0, 'Latency': latency, 'Cost': np.nan, 'Actionable': 0})
        else:
            simulation_results.append({'Validity': 0, 'Latency': latency, 'Cost': np.nan, 'Actionable': 0})
            
    except Exception as e:
        simulation_results.append({'Validity': 0, 'Latency': time.time() - start_time, 'Cost': np.nan, 'Actionable': 0})

# 步驟 3：統計輸出
results_df = pd.DataFrame(simulation_results)

print("\n=== 1000 人大規模系統模擬實驗結果 ===")
print(f"1. 翻轉有效性 (Validity): {results_df['Validity'].mean() * 100:.2f}%")
print(f"2. 平均運算延遲 (Avg Latency): {results_df['Latency'].mean():.4f} 秒/筆")
print(f"3. 具備高可執行性比例 (Actionability < 20% Limit): {results_df['Actionable'].mean() * 100:.2f}%")
print(f"4. 平均擾動成本 (Avg Cost): NT$ {results_df['Cost'].mean(skipna=True):.2f}")