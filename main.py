from operator import itemgetter
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.chat_models.gigachat import GigaChat
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from secret import auth


def get_report(task: str, user_text: str, score: int = 1) -> str:
    llm = GigaChat(
        profanity_check=False,
        verify_ssl_certs=False,
        credentials=auth,
        temperature=0.1,
    )
    validate_chain = LLMChain(
        llm=llm,
        prompt=load_prompt("src/prompt/validate.yaml"),
        output_key="report",
    )
    chain = {
        "task": itemgetter("task"),
        "user_text": itemgetter("user_text"),
        "score": itemgetter("score"),
    } | RunnablePassthrough() | validate_chain

    report = chain.invoke({
        "task": task,
        "user_text": user_text,
        "score": score
    })['report']

    return report


if __name__ == "__main__":
    st.header("Автоматическая проверка критериев")
    st.markdown("<hr>", unsafe_allow_html=True)
    df = pd.read_excel("src/Новая таблица-2.xlsx")
    df = df[['Цена ошибки', "Критерии"]]
    st.subheader("Критерии, которые будут проверяться:")
    for element in df['Критерии']:
        st.markdown(f"- {element}")

    text = st.text_area("Введите текст для проверки")
    if text:
        df['Цена ошибки'] = df.apply(lambda row: get_report(task=row['Критерии'],
                                                            user_text=text,
                                                            score=row['Цена ошибки']), axis=1)
        df['predict_bool'] = df['Цена ошибки'].apply(lambda x: False if x =='0' else True)
        df['Цена ошибки']  = df['Цена ошибки'].astype(int)

        st.dataframe(df, height=450)
        st.write(f"Общая сумма ошибок: {df['Цена ошибки'].sum()}")