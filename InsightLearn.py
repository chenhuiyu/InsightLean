import streamlit as st
from langchain import OpenAI

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGUAGE_INSTRUCTIONS_DICT = {
    "Chinese": "- 请使用中文输出\n",
    "English": "- Please output English.\n"
}

def load_LLM():
    llm = OpenAI(temperature=0.5)
    return llm

def get_language_instructions(language):
    return LANGUAGE_INSTRUCTIONS_DICT.get(language, "")

def handle_translation(llm, input_text, language):
    template = """
    Below is the target output language:
    - Language: {language}
    Below is the input text.
    - Text: {text}

    Your goal is to generate output in the corresponding language based on the input text.Usually use a professional tone in academic papers, and the fields of the papers include civil engineering and mechanics:
    - Output Text:
    """
    prompt_with_query = template.format(language=language, text=input_text)

    if input_text:
        output = llm(prompt_with_query)
        st.markdown("### Your Translated Text")
        st.success(output)

def main():
    st.set_page_config(page_title="Paper Translator", page_icon=":robot:")
    st.header("Chinese-to-English Paper Translator")
    input_text = st.text_area(label="", placeholder="请在此处输入中文论文文本...", key="text_input")
    llm = load_LLM()
    if input_text and st.button("Translate!"):
        language = get_language_instructions("English")
        handle_translation(llm, input_text, language)

if __name__ == '__main__':
    main()
