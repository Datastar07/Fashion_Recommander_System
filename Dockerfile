FROM python

WORKDIR /main

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . /main

# ENTRYPOINT ["streamlit","run"]

CMD ["streamlit","run","streamlit_app.py"]