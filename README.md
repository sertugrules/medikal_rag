# Proje Kurulum ve Çalıştırma Rehberi

## 1. Reponun Klonlanması

git clone https://github.com/kullaniciAdi/projeAdi.git
cd projeAdi


## VSCODE üzerinde
### pip install pipenv
### pipenv install gerekli importları eder
### pipenv shell

## .env dosyası oluşturulur api keyler yapıştırılır. 
### Mailde ileteceğim ama token error gelirse üyelik açıp yapıştırmanız gerekebilir 
### HF_TOKEN = https://huggingface.co/models
### LANGCHAIN_TOKEN = https://www.langchain.com/
### GROQ_API_KEY = https://groq.com/

Seçenek 1: Google Drive’dan Manuel Kopyalama
vectorstore/ klasörünün içine aşağıdaki dosyaları yerleştirin:

index.faiss

index.pkl

Gerekirse data/ klasörüne PDFleri koyun

Seçenek 2: Kendi Verinle Vektör Oluşturma
data/ klasörüne PDF koyulur

Aşağıdaki komutu çalıştırarak vektörleri oluşturun:
pipenv run python vector_builder.py

# Uygulamanın başlatılması
pipenv shell
uvicorn main:app --reload

# Ön yüzünün başlatılması
streamlit run app.py


# Docker çalıştırılması
## .env klasörünün boş olmamasına dikkat edin
## docker-compose build
## docker-compose up
