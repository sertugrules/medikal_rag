import streamlit as st
import requests

st.set_page_config(page_title="Medikal RAG Chatbot", layout="centered")
st.title("Medikal RAG Chatbot")
st.markdown("Aşağıya medikal sorunuzu yazın:")

# Model seçimi
models = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]
selected_model = st.selectbox("Kullanmak istediğiniz modeli seçin:", models)

API_URL = "http://127.0.0.1:9999"

# Varsayılan session_state ayarları
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "latency" not in st.session_state:
    st.session_state.latency = ""
if "query" not in st.session_state:
    st.session_state.query = ""
if "rel" not in st.session_state:
    st.session_state.rel = 3
if "acc" not in st.session_state:
    st.session_state.acc = 3
if "flu" not in st.session_state:
    st.session_state.flu = 3
if "src_flag" not in st.session_state:
    st.session_state.src_flag = True

# Soru alanı
user_query = st.text_area(
    "Soru",
    height=150,
    placeholder="Örneğin: Diyabetin belirtileri nelerdir?"
)

# Sorgu gönderme
if st.button("Soruyu Gönder"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "messages": [user_query]
        }

        try:
            response = requests.post(f"{API_URL}/query", json=payload)
            response_json = response.json()

            if response.status_code != 200 or "error" in response_json:
                st.error("Hata: " + response_json.get("error", "Bilinmeyen bir hata oluştu."))
            else:
                st.session_state.answer = response_json["answer"]
                st.session_state.sources = response_json["sources"]
                st.session_state.latency = response_json["latency"]
                st.session_state.query = user_query

        except Exception as e:
            st.error(f"Sunucuya istek gönderilirken hata oluştu: {e}")

# Yanıt gösterimi
if st.session_state.answer:
    st.subheader("Cevap")
    st.markdown(st.session_state.answer)

    st.subheader("Kaynaklar")
    for src in st.session_state.sources:
        st.code(src)

    st.subheader("Yanıt Süresi")
    st.markdown(st.session_state.latency)

    # Değerlendirme
    st.subheader("Cevabı Değerlendirin")

    st.slider("İlgililik (1-5)", 1, 5, st.session_state.rel, key="rel")
    st.slider("Doğruluk (1-5)", 1, 5, st.session_state.acc, key="acc")
    st.slider("Akıcılık (1-5)", 1, 5, st.session_state.flu, key="flu")
    st.checkbox("Kaynaklar yeterli ve doğru mu?", value=st.session_state.src_flag, key="src_flag")

    def submit_evaluation():
        eval_payload = {
            "query": st.session_state.query,
            "relevance": st.session_state.rel,
            "accuracy": st.session_state.acc,
            "fluency": st.session_state.flu,
            "sources_flag": st.session_state.src_flag
        }
        try:
            eval_response = requests.post(f"{API_URL}/manual_evaluation", json=eval_payload)
            if eval_response.status_code == 200:
                st.success("Değerlendirmeniz kaydedildi.")
            else:
                st.error("Değerlendirme kaydedilirken hata oluştu.")
        except Exception as e:
            st.error(f"Değerlendirme gönderilemedi: {e}")

    st.button("Değerlendirmeyi Gönder", on_click=submit_evaluation)
