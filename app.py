# ============================================================
# app.py — v17.0 (修正 APIResponse.error 兼容性問題)
# ============================================================

import os
import io
import re
import json 
import base64 
import time
from datetime import datetime
from uuid import uuid4 # 用於產生唯一的 ID

import streamlit as st
from PIL import Image
# 移除 psycopg2
import pandas as pd
import pytesseract
import plotly.express as px

# --- Supabase 依賴 ---
try:
    from supabase import create_client, Client
    # 導入 APIError 以便捕獲錯誤
    from postgrest.exceptions import APIError 
except ImportError:
    st.error("請安裝 supabase 函式庫: pip install supabase")
    st.stop()
# --- Supabase 依賴 ---


# 導入 UNet 相關
try:
    from inference import run_unet_inference
    from inference import visualize_mask
except ImportError:
    # 如果 inference 模組不存在，提供空函式避免程式崩潰
    def run_unet_inference(pil_img, checkpoint_path):
        # 回傳 None, None, Empty dict
        return None, None, {} 
    def visualize_mask(mask):
        return Image.new('RGB', (100, 100), color = 'red')

from openai import OpenAI

# ------------------------------------------------------------
# 1. 自動偵測 Tesseract.exe（Windows）
# ------------------------------------------------------------
def auto_set_tesseract_path():
    """自動偵測 Tesseract OCR 執行檔路徑"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None

TESSERACT_PATH = auto_set_tesseract_path()


# ------------------------------------------------------------
# 2. Supabase / PostgreSQL 設定
# ------------------------------------------------------------
# 🚨 請在這裡填入您的 Supabase 專案資訊
SUPABASE_URL = "https://tervudnniyobpeancuhj.supabase.co" # 替換為您的專案 URL
# 使用 Service Role Key 進行後端操作(自行輸入
SERVICE_ROLE_KEY = " " 
TABLE_NAME = "invoices_data" # 確保此名稱與您在 Supabase 中建立的表格名稱完全一致

@st.cache_resource
def get_supabase_client():
    """初始化並回傳 Supabase 客戶端"""
    if not SERVICE_ROLE_KEY or SERVICE_ROLE_KEY == "您的 Service Role Key (sb_secret_...)":
        # 這裡的檢查現在應該不會觸發，因為 Service Key 已經填入
        st.error("🚨 警告：請在 app.py 檔案中填入有效的 SUPABASE_URL 和 SERVICE_ROLE_KEY！")
        return None
        
    try:
        supabase: Client = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)
        return supabase
    except Exception as e:
        st.error(f"Supabase 連線失敗: {e}")
        return None

# 取得 Supabase 客戶端實例
supabase = get_supabase_client()


# ------------------------------------------------------------
# 3. OpenAI 配置
# ------------------------------------------------------------
# 🚨 修正：請在這裡填入您有效的 API Key
client = OpenAI(api_key=" ")

# ------------------------------------------------------------
# 4. 常數
# ------------------------------------------------------------
CATEGORIES = ["餐飲", "交通", "購物", "娛樂", "醫療", "教育", "雜項"]
CHECKPOINT_PATH = "checkpoints/unet_epoch30.pth" # 假設您的模型在這裡

# ------------------------------------------------------------
# 5. 函數：LLM 驗證與修正
# (此函數無變動)
# ------------------------------------------------------------

def llm_validate_and_correct(img_bytes, ocr_results, user_query):
    """使用 GPT-4-Vision 進行 OCR 結果驗證與修正"""
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    prompt = f"""
    您是一位專業的發票資料審核員。您面前有一張發票圖片和初步的 OCR 辨識結果。
    
    **OCR 結果:**
    發票號碼: {ocr_results.get('invoice_no', 'N/A')}
    日期: {ocr_results.get('date', 'N/A')}
    金額: {ocr_results.get('total_amount', 'N/A')}
    
    **任務:**
    1. **檢查**圖片，特別是 OCR 辨識出來的**發票號碼**、**日期**和**總金額**是否正確。
    2. **修正**任何錯誤，並以 **JSON** 格式回傳最終結果。JSON 必須包含 "發票號碼"、"日期" 和 "金額" 三個鍵。
    3. 如果某個欄位無法辨識，請填寫 `"N/A"`。
    
    **用戶額外請求:** {user_query}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        llm_output = json.loads(response.choices[0].message.content)
        return llm_output
    except Exception as e:
        print(f"LLM 呼叫或解析錯誤: {e}")
        return None


# ------------------------------------------------------------
# 6. 函數：資料儲存 (使用 Supabase)
# ------------------------------------------------------------

def save_invoice(img_bytes, data):
    # 確保 Supabase 客戶端已初始化
    if supabase is None:
        st.error("資料庫服務未初始化，無法儲存。")
        return
    
    # 將圖片轉換為 Base64 字串
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    try:
        # 準備要插入的單筆紀錄
        record = {
            "invoice_id": str(uuid4()), # 生成新的 UUID
            "invoice_no": data.get("invoice_no"),
            "date": data.get("date"),
            "total_amount": float(data.get("total_amount")),
            "category": data.get("category"),
            "note": data.get("note", "無"), 
            "created_at": datetime.now().isoformat(),
            "image_base64": img_base64
        }
        
        # 執行插入操作
        response = supabase.table(TABLE_NAME).insert(record).execute()
        
        # 關鍵修正：檢查 response.data 是否包含數據來判斷是否成功
        if response.data is not None and len(response.data) > 0:
            st.success(f"✔ 資料已寫入 Supabase，Invoice ID={response.data[0].get('invoice_id', 'N/A')}")
        else:
            # 如果 data 是空列表，通常代表操作失敗或沒有任何行被影響
            st.error("寫入 Supabase 失敗：資料庫回傳無紀錄或操作失敗。")
            
    except APIError as e:
        # 如果是 APIError，則可以直接顯示其訊息
        st.error(f"寫入 Supabase 失敗 (APIError): {e.code} - {e.message}")
    except Exception as e:
        st.error(f"寫入 Supabase 發生未預期錯誤: {e}")


# ------------------------------------------------------------
# 7. 函數：資料查詢 (使用 Supabase)
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data_for_dashboard():
    # 確保 Supabase 客戶端已初始化
    if supabase is None:
        return pd.DataFrame()

    try:
        # 執行查詢操作
        response = supabase.table(TABLE_NAME).select(
            "invoice_id, invoice_no, date, total_amount, category, note, created_at"
        ).order(
            "created_at", desc=True
        ).execute()
        
        # 關鍵修正：檢查 response.data 是否為 None 或空列表
        if response.data is None or len(response.data) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(response.data)
        
    except APIError as e:
        # 如果是 APIError，則可以直接顯示其訊息
        st.error(f"Supabase 讀取失敗 (APIError): {e.code} - {e.message}")
        return pd.DataFrame()
    except Exception as e:
        # 處理任何其他意外錯誤
        st.error(f"Supabase 讀取發生未預期錯誤: {e}")
        return pd.DataFrame()
    
    
    if len(df) > 0:
        # 將 'invoice_id' 重新命名為 'id' 以兼容儀表板邏輯 (如果需要，但此處使用 Supabase 欄位名更清晰)
        df.rename(columns={'invoice_id': 'id'}, inplace=True)
        
        # 數據清洗與轉換
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        df = df.dropna(subset=['date', 'total_amount'])
        df['YearMonth'] = df['date'].dt.to_period('M')
        
    return df

# ------------------------------------------------------------
# 8. Streamlit 主體
# (主體程式碼無變動)
# ------------------------------------------------------------

st.set_page_config(
    page_title="智能發票記帳神器",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 智能發票記帳神器")

# API Key 側邊欄輸入 
with st.sidebar:
    st.header("🔑 偵測與配置")
    openai_key = st.text_input("OpenAI API Key (gpt-4o)", type="password", help="用於 LLM 驗證與修正")
    
    if openai_key:
        client.api_key = openai_key
        st.success("OpenAI Key 已配置")
    else:
        st.warning("請在側邊欄輸入您的 OpenAI Key")
    
    if TESSERACT_PATH:
        st.info(f"Tesseract OCR 已偵測: {TESSERACT_PATH}")
    else:
        st.error("Tesseract OCR 未偵測到。請檢查路徑或安裝。")

# --- Tabs ---
tab1, tab2 = st.tabs(["🧾 掃描與記錄", "📊 分析與紀錄"])

# ========== TAB 1：掃描與記錄 ==========
with tab1:
    st.header("發票掃描與 AI 辨識")

    col1_upload, col2_control = st.columns([1, 2])
    
    uploaded = col1_upload.file_uploader(
        "**請上傳發票圖片 (JPG/PNG)**", 
        type=["jpg", "png", "jpeg"],
        help="建議圖片清晰、對焦良好"
    )

    # 執行辨識按鈕
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        
    process_button = col2_control.button(
        "🧠 啟動 AI 辨識", 
        type="secondary",
        disabled=uploaded is None or st.session_state.processing
    )
    
    # 狀態初始化
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {
            "inv_no": "N/A",
            "parsed_date": "N/A",
            "amount": 0,
            "pil_img": None # 初始為 None
        }

    if uploaded and process_button:
        st.session_state.processing = True
        
        with st.spinner("🚀 AI 辨識中 (UNet Segmentation -> Tesseract OCR -> GPT-4o 驗證)..."):
            
            # --- 影像載入與準備 ---
            img_bytes = uploaded.getvalue()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # --- 1. UNet Segmentation + Bounding Box ---
            try:
                mask, bboxes, crops_map = run_unet_inference(pil_img, CHECKPOINT_PATH)
            except Exception as e:
                st.error(f"UNet 推論失敗: {e}")
                st.session_state.processing = False
                st.session_state.current_data["pil_img"] = pil_img
                st.stop()
                
            
            # --- 2. Tesseract OCR ---
            ocr_results = {}
            for field, cropped_img in crops_map.items():
                if cropped_img:
                    ocr_text = pytesseract.image_to_string(cropped_img, lang='eng', config='--psm 6').strip()
                    ocr_results[field] = ocr_text.replace('\n', ' ')
            
            # --- 3. LLM 驗證 ---
            if openai_key:
                llm_output = llm_validate_and_correct(img_bytes, ocr_results, "請確保總金額為數字")
                
                if llm_output:
                    inv_no = llm_output.get("發票號碼", "N/A")
                    parsed_date = llm_output.get("日期", "N/A")
                    amount_str = str(llm_output.get("金額", "0")).replace(',', '').strip()
                    try:
                        amount = float(re.sub(r'[^\d.]', '', amount_str))
                    except ValueError:
                        amount = "N/A"
                else:
                    st.error("LLM 驗證失敗，請手動修正資料。")
                    inv_no, parsed_date, amount = "N/A", "N/A", "N/A" # LLM 失敗時給予 N/A
            else:
                # 無 Key 狀態下，使用基礎 OCR 結果
                inv_no = ocr_results.get('invoice_no', 'N/A')
                parsed_date = ocr_results.get('date', 'N/A')
                amount_str = ocr_results.get('total_amount', '0').replace(',', '').strip()
                try:
                    amount = float(re.sub(r'[^\d.]', '', amount_str))
                except ValueError:
                    amount = "N/A"
            
            st.session_state.processing = False
            # 儲存結果到 session_state
            st.session_state.current_data = {
                "inv_no": inv_no,
                "parsed_date": parsed_date,
                "amount": amount,
                "pil_img": pil_img
            }
            # 重新運行以顯示結果
            st.rerun()

    # --- 顯示圖片與結果 ---
    # 只要上傳了圖片，或者 session_state 中有圖片數據，就進入顯示區塊
    if uploaded or st.session_state.current_data["pil_img"] is not None:
        
        # 確保當 uploaded 存在但 current_data["pil_img"] 為 None 時，使用 uploaded 的圖片
        if uploaded and st.session_state.current_data["pil_img"] is None:
            img_bytes = uploaded.getvalue()
            st.session_state.current_data["pil_img"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 從 session_state 讀取資料
        inv_no = st.session_state.current_data["inv_no"]
        parsed_date = st.session_state.current_data["parsed_date"]
        amount = st.session_state.current_data["amount"]
        pil_img = st.session_state.current_data["pil_img"]
        
        
        # UI 分割
        col1_img, col2_input = st.columns([3, 2])
        
        with col1_img:
            st.subheader("🖼️ 發票圖片預覽")
            # 修正點 1: use_column_width=True -> use_container_width=True
            st.image(pil_img, caption="原始發票圖片", use_container_width=True) 

        with col2_input:
            st.subheader("📝 確認與分類")
            
            # --- 辨識結果 ---
            st.metric("發票號碼", inv_no)
            st.metric("日期", parsed_date)

            # 手動修正金額
            current_amount = amount if isinstance(amount, (int, float)) else 0
            
            st.metric("AI 辨識金額", f"NT${current_amount:,.0f}" if isinstance(amount, (int, float)) else str(amount))
            
            final_amount = st.number_input(
                "手動修正金額", 
                min_value=0, 
                max_value=500000, 
                value=int(current_amount),
                step=1
            )

            # --- 分類與儲存控制 ---
            st.markdown("---")
            category = st.selectbox("消費類別", CATEGORIES)
            note = st.text_input("項目/備註", "")

            # ===== 儲存按鈕 =====
            is_valid = (
                isinstance(inv_no, str) and inv_no != "N/A" and 
                isinstance(parsed_date, str) and parsed_date != "N/A" and 
                final_amount > 0
            )
            
            if st.button("💾 確認儲存至資料庫", type="primary", disabled=not is_valid):
                if not is_valid:
                    st.error("資料無效 (發票號碼/日期/金額)，無法儲存。")
                else:
                    data = {
                        "invoice_no": inv_no,
                        "date": parsed_date,
                        "total_amount": final_amount,
                        "category": category,
                        "note": note
                    }
                    
                    img_bytes_io = io.BytesIO()
                    st.session_state.current_data["pil_img"].save(img_bytes_io, format='JPEG')
                    
                    save_invoice(img_bytes_io.getvalue(), data)
                    
                    # 儲存後清除並刷新儀表板資料緩存
                    st.cache_data.clear() 
                    # 清除 current_data 以避免重複儲存
                    st.session_state.current_data = {
                        "inv_no": "N/A", "parsed_date": "N/A", "amount": 0, "pil_img": None
                    }
                    st.rerun()
                    
    elif uploaded:
        st.info("點擊 '🧠 啟動 AI 辨識' 開始處理。")


# ========== TAB 2：分析與紀錄 (使用修正後的 load_data_for_dashboard) ==========
with tab2:
    st.header("📈 記帳分析儀表板")

    # 呼叫緩存函數，確保連線在函數內被管理和關閉
    df = load_data_for_dashboard()
    
    if len(df) == 0:
        st.info("尚無發票紀錄，請先到「掃描與記錄」分頁新增資料。")
        # 僅在 Supabase 連線成功但無數據時顯示此資訊
        if supabase is not None:
             st.stop()
        else:
             # 如果連線失敗，讓 Streamlit 繼續執行以顯示錯誤信息
             pass


    # --- 1. 總覽 KPI ---
    total_spending = df['total_amount'].sum()
    st.subheader(f"總結 ({df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')})")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    col_kpi1.metric("總消費筆數", f"{len(df):,}")
    col_kpi2.metric("總消費金額", f"NT${total_spending:,.0f}")
    
    # 計算最近一個月的總支出
    # 處理 Period 類型比較
    if not df['YearMonth'].empty:
        latest_month_period = df['YearMonth'].max()
        df_latest_month = df[df['YearMonth'] == latest_month_period] 
        monthly_spending = df_latest_month['total_amount'].sum()
        col_kpi3.metric(f"{latest_month_period.strftime('%Y 年 %m 月')} 總開銷", f"NT${monthly_spending:,.0f}")
    
    st.markdown("---")


    # --- 2. 視覺化分析區 ---
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("💸 消費類別佔比")
        
        # 排除收入類別
        df_expense = df[df['category'] != '收入']
        category_summary = df_expense.groupby('category')['total_amount'].sum().reset_index()
        
        if len(category_summary) > 0:
            fig_pie = px.pie(
                category_summary,
                values='total_amount',
                names='category',
                title='各類別支出分佈',
                hole=.3, # 甜甜圈圖
                color_discrete_sequence=px.colors.qualitative.T10 
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            fig_pie.update_layout(showlegend=False)
            # 修正點 2: use_container_width=True
            st.plotly_chart(fig_pie, use_container_width=True) 
        else:
            st.info("暫無支出數據可供分析。")

    with col_chart2:
        st.subheader("📊 月度支出趨勢")
        
        # 按月度計算總和 (排除收入)
        monthly_trend = df_expense.groupby('YearMonth')['total_amount'].sum().reset_index()
        monthly_trend['Month'] = monthly_trend['YearMonth'].astype(str)
        
        if len(monthly_trend) > 0:
            fig_line = px.line(
                monthly_trend.sort_values(by='Month'), 
                x='Month',
                y='total_amount',
                title='月度支出總額趨勢',
                labels={'total_amount': '支出金額 (NT$)', 'Month': '月份'},
                markers=True,
                color_discrete_sequence=['#4c78a8']
            )
            fig_line.update_traces(line=dict(width=3))
            fig_line.update_layout(xaxis_tickangle=-45)
            # 修正點 3: use_container_width=True
            st.plotly_chart(fig_line, use_container_width=True) 
        else:
            st.info("暫無歷史數據可供分析。")
            
    st.markdown("---")
    
    # --- 3. 歷史紀錄表格 (優化顯示) ---
    st.subheader("🧾 歷史帳目明細 (依月份整理)")
    
    pivot = df.sort_values(by=['date', 'created_at'], ascending=[False, False])
    
    display_cols = ['date', 'invoice_no', 'total_amount', 'category', 'note']
    display_name_map = {"date": "消費日期", "invoice_no": "發票號碼", "total_amount": "總金額 (NT$)", "category": "類別", "note": "項目/備註"}

    for period, group in pivot.groupby('YearMonth', sort=False):
        monthly_total = group['total_amount'].sum()
        
        with st.expander(f"📅 **{period.strftime('%Y 年 %m 月')}** — 總消費：NT${monthly_total:,.0f}", expanded=False):
            
            month_df = group[display_cols].rename(columns=display_name_map)
            month_df['消費日期'] = month_df['消費日期'].dt.strftime('%Y-%m-%d')
            
            month_df['總金額 (NT$)'] = month_df['總金額 (NT$)'].apply(lambda x: f"{x:,.0f}")
            
            # 修正點 4: use_container_width=True
            st.dataframe(
                month_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "總金額 (NT$)": st.column_config.TextColumn(
                        "總金額 (NT$)",
                        help="本筆消費金額",
                        disabled=True
                    )
                }
            )


