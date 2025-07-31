### 1. Reponun Klonlanması
# .env Dosyasının Oluşturulması
Proje dizinine .env adında bir dosya oluşturun ve aşağıdaki içeriği yapıştırın:
ilgili mailde ilettiğim
HF_TOKEN=
LANGCHAIN_TOKEN=
GROQ_API_KEY=


## VSCODE üzerinde
### pip install pipenv
### pipenv install gerekli importları eder
### pipenv shell

Seçenek 1: Google Drive’dan Manuel Kopyalama
vectorstore/ klasörünün içine aşağıdaki dosyaları yerleştirin:

index.faiss

index.pkl

Gerekirse data/ klasörüne PDF veya metin dosyalarını da ekleyebilirsiniz.

Seçenek 2: Kendi Verinle Vektör Oluşturma
data/ klasörüne PDF veya .txt belgelerini koyun.

Aşağıdaki komutu çalıştırarak vektörleri oluşturun:
pipenv run python vector_builder.py

# Uygulamanın başlatılması
pipenv shell
uvicorn main:app --reload

# Ön yüzünün başlatılması
streamlit run app.py
