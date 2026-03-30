import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="My AI Portfolio", page_icon="✨", layout="wide")

@st.cache_resource
def load_mortality_model():
    return joblib.load('mortality_ensemble_model.pkl')

@st.cache_data
def load_mortality_data():
    return pd.read_csv('cleaned_mortality_data.csv')

@st.cache_resource
def load_dog_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None, 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.load_weights('best_dog_model.keras')
    return model

# ตัวแปรสายพันธุ์สุนัข
CLASS_NAMES = [
    'Beagle', 'Chihuahua', 'Corgi', 'Doberman', 'Golden Retriever',
    'Labrador', 'Pit Bull', 'Shiba Inu', 'Shih-Tzu', 'Siberian Husky'
]

st.sidebar.title("เมนูหลัก (Navigation)")
page = st.sidebar.radio("เลือกแอปพลิเคชันที่ต้องการใช้งาน:", 
                        ["📈 พยากรณ์จำนวนผู้เสียชีวิต", "🐶 AI แยกสายพันธุ์สุนัข"])


if page == "📈 พยากรณ์จำนวนผู้เสียชีวิต":
    
    model = load_mortality_model()
    df_history = load_mortality_data()

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .section-label {
        font-size: 11px; font-weight: 500; letter-spacing: 0.14em;
        text-transform: uppercase; color: #aaa; margin-bottom: 20px;
        margin-top: 48px; display: flex; align-items: center; gap: 12px;
    }
    .section-label::after { content: ''; flex: 1; height: 1px; background: #e0ddd8; }
    .result-card {
        border: 1px solid #e8e5df; border-radius: 4px; padding: 28px 32px;
        background: #fff; display: flex; align-items: flex-start;
        justify-content: space-between; gap: 24px; margin-top: 20px;
    }
    .result-card .label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #666; margin-bottom: 8px; }
    .result-card .value { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 400; color: #1a1a1a; line-height: 1; }
    .result-card .meta { font-size: 12px; color: #555; margin-top: 10px; font-weight: 300; }
    .result-card .tag { font-family: 'DM Mono', monospace; font-size: 12px; color: #333; background: #f3f1ec; padding: 6px 12px; border-radius: 2px; white-space: nowrap; align-self: center; }
    .footnote { font-size: 11px; color: #888; font-weight: 300; margin-top: 12px; letter-spacing: 0.02em; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Mortality Forecast")
    st.markdown("พ.ศ. 2561 – 2569 · ประเทศไทย")

    # Section 1
    st.markdown('<div class="section-label">01 · แนวโน้มรายปี</div>', unsafe_allow_html=True)
    yearly_deaths = df_history.groupby('Year')['Deaths'].sum().reset_index()
    fig1 = px.line(yearly_deaths, x='Year', y='Deaths', markers=True, labels={'Year': '', 'Deaths': ''})
    fig1.update_traces(line=dict(color='#1a1a1a', width=1.5), marker=dict(size=5, color='#1a1a1a', symbol='circle'))
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=24, r=24, t=16, b=24), height=240, hovermode='x unified')
    
    with st.container(border=True):
        st.plotly_chart(fig1, use_container_width=True)

    # Section 2
    st.markdown('<div class="section-label">02 · การกระจายตัวตามอายุและเพศ</div>', unsafe_allow_html=True)
    age_gender_deaths = df_history.groupby(['Age', 'Gender'])['Deaths'].sum().reset_index()
    age_gender_deaths['เพศ'] = age_gender_deaths['Gender'].map({0: 'ชาย', 1: 'หญิง'})
    fig2 = px.line(age_gender_deaths, x='Age', y='Deaths', color='เพศ', labels={'Age': '', 'Deaths': ''}, color_discrete_map={'ชาย': '#2563c8', 'หญิง': '#e0457b'})
    fig2.update_traces(line=dict(width=2))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=24, r=24, t=16, b=24), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), height=240, hovermode='x unified')
    
    with st.container(border=True):
        st.plotly_chart(fig2, use_container_width=True)

    # Section 3
    st.markdown('<div class="section-label">03 · พยากรณ์ปี พ.ศ. 2569</div>', unsafe_allow_html=True)
    

    with st.container(border=True):
        st.markdown(
            '<p style="font-size: 13px; color: #666; font-weight: 300; line-height: 1.7; margin-bottom: 24px;">'
            'ระบุกลุ่มประชากรที่ต้องการพยากรณ์ — โมเดลจะประมาณจำนวนผู้เสียชีวิตในปี พ.ศ. 2569 ล่วงหน้า 1 ปี</p>', 
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            user_age = st.number_input("อายุ (ปี)", min_value=0, max_value=101, value=30, step=1)
        with col2:
            user_gender_text = st.selectbox("เพศ", ["ชาย", "หญิง"])
        with col3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run = st.button("พยากรณ์ →", use_container_width=True, type="primary")

    if run:
        user_gender_val = 0 if user_gender_text == "ชาย" else 1
        input_data = pd.DataFrame([[2569, user_age, user_gender_val]], columns=['Year', 'Age', 'Gender'])
        prediction = model.predict(input_data)
        predicted_deaths = int(round(prediction[0]))

        st.markdown(f"""
        <div class="result-card">
            <div>
                <div class="label">คาดการณ์จำนวนผู้เสียชีวิต</div>
                <div class="value">{predicted_deaths:,}</div>
                <div class="meta">คน · ปี พ.ศ. 2569</div>
            </div>
            <div class="tag">เพศ{user_gender_text} · อายุ {user_age} ปี</div>
        </div>
        <p class="footnote">Ensemble Learning — Random Forest + SVR + Gradient Boosting</p>
        """, unsafe_allow_html=True)


elif page == "🐶 AI แยกสายพันธุ์สุนัข":
    
    # โหลดโมเดล
    dog_model = load_dog_model()

    st.title("AI แยกสายพันธุ์สุนัข")
    st.write("อัปโหลดรูปภาพน้องหมาของคุณ แล้วให้ AI ของเราช่วยทายสายพันธุ์ดูสิ!")

    st.markdown("### 📌 10 สายพันธุ์ที่ AI ของเรารู้จัก")

    cols = st.columns(5)
    for i, breed in enumerate(CLASS_NAMES):
        col_index = i % 5 
        with cols[col_index]:
            filename = breed.lower().replace(' ', '_')
            image_path = f"C:/6704062617148_Ptoject_IS/images/{filename}.jpg" 
            
            try:
                st.image(image_path, use_container_width=True)
            except FileNotFoundError:
                st.error(f"หารูป {filename}.jpg ไม่พบ")
                    
            st.markdown(f"<p style='text-align: center;'><b>{breed}</b></p>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("กรุณาอัปโหลดรูปภาพสุนัข")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        

        with st.container(border=True):
            st.image(image, caption="รูปภาพที่คุณอัปโหลด", use_container_width=True)
        
        with st.spinner('AI กำลังวิเคราะห์รูปภาพ... 🕵️‍♂️'):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_expanded = np.expand_dims(img_array, axis=0)
            
            predictions = dog_model.predict(img_expanded)[0]
            top_3_indices = predictions.argsort()[-3:][::-1]
            
            st.success(f"🎉 AI มั่นใจว่านี่คือ **{CLASS_NAMES[top_3_indices[0]]}**")
            
            st.markdown("### รายละเอียดการวิเคราะห์ (Top 3)")
            with st.container(border=True):
                for idx in top_3_indices:
                    breed = CLASS_NAMES[idx]
                    confidence = predictions[idx] * 100
                    
                    text_col, bar_col = st.columns([1, 3])
                    text_col.write(f"**{breed}**")
                    bar_col.progress(int(confidence), text=f"{confidence:.2f}%")

    st.markdown("---")
    st.subheader("การเตรียมข้อมูล (Data Preparation)")
    st.write("- ใช้Dataset จาก https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set")
    st.write("- ข้อมูลรูปภาพถูกแบ่งออกเป็น 2 ชุดหลักคือ ชุดฝึกสอน (Training Set) และ ชุดตรวจสอบ (Validation Set)")
    st.write("- การปรับขนาดภาพ (Image Resizing): รูปภาพทั้งหมดจะถูกปรับขนาดให้เป็น 224x224 พิกเซล เพื่อให้สอดคล้องกับขนาด Input มาตรฐานที่โมเดล EfficientNetB0 ต้องการ")
    st.write("- เราใช้ ImageDataGenerator ในการสร้างความหลากหลายให้ชุดข้อมูล Train เช่น การหมุนภาพ (Rotation), การซูม (Zoom), การเลื่อนภาพ (Shift), และการพลิกซ้ายขวา (Horizontal Flip) ทำให้โมเดลเรียนรู้โครงสร้างของสุนัขได้ดีขึ้นในหลายๆ มุมมอง")

    st.subheader("ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm & Theory)")
    st.write("- CNN - Convolutional Neural Network")
    st.write("- Efficientnet Architecture")
    st.write("- Transfer Learning")

    st.subheader("ขั้นตอนการพัฒนาโมเดล (Model Development Steps)")
    st.write("- Base Model: นำโมเดล EfficientNetB0 มาตัดเลเยอร์ส่วนหัวทิ้ง และทำการFreeze ค่าน้ำหนักเดิมไว้เพื่อรักษาความรู้พื้นฐานของ AI")
    st.write("- สร้างส่วนตัดสินใจใหม่: นำโมเดลมาต่อเติมเลเยอร์ใหม่ (เช่น Pooling, Dropout, Dense) โดยให้ชั้นสุดท้ายมี 10 โหนด เพื่อทายผล 10 สายพันธุ์")
    st.write("- Compile: กำหนดให้โมเดลใช้ Optimizer แบบ Adam และใช้ Loss Function แบบ Categorical Crossentropy สำหรับงานแยกหลายคลาส")
    st.write("- ฝึกสอนโมเดล (Train & Callbacks): เริ่มการเทรนโดยใช้ตัวช่วยเพื่อหยุดเทรนอัตโนมัติหากโมเดลไม่พัฒนา และตั้งให้บันทึกเฉพาะโมเดลที่แม่นยำที่สุดเท่านั้น")
    st.write("- (Deploy): นำไฟล์โมเดลที่ดีที่สุด (.keras) ไปเขียนโค้ดด้วย Streamlit เพื่อสร้าง Web App ให้ผู้ใช้อัปโหลดรูปภาพมาทดสอบได้จริง")