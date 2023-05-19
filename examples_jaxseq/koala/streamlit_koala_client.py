import tyro
import streamlit as st
from typing import Optional, Tuple, List, Union
from examples_jaxseq.koala.commandline_koala_client import KoalaChatClient
import json
from JaxSeq.stream_tokens import map_token_stream

def main(
    host: Union[str, List[str]]='http://127.0.0.1:8000/', 
    max_input_length: int=1024, 
    max_output_length: int=1024, 
    model_name: Optional[str]='Koala', 
    eos_token_id: Optional[int]=None, 
    sep_turn_text: str='</s>', 
    stream: bool=True, 
):
    if isinstance(host, str):
        host = [host]
    if 'client' not in st.session_state:
        st.session_state['client'] = KoalaChatClient(hosts=host, sep_turn_text=sep_turn_text)
    client = st.session_state['client']

    def format_chat_text(curr_response: Optional[str]):
        if curr_response is None:
            assert len(client.user_messages) == len(client.model_responses)
        else:
            assert len(client.user_messages) == (len(client.model_responses)+1)
        
        text = ''
        for user_message, model_response in zip(client.user_messages, client.model_responses):
            text += f'<span style="color:lightcoral">**User:**</span> {user_message}\n<span style="color:lightblue">**Koala:**</span> {model_response}\n'
        
        if curr_response is not None:
            text += f'<span style="color:lightcoral">**User:**</span> {client.user_messages[-1]}\n<span style="color:lightblue">**Koala:**</span> {curr_response}'
        return text

    if model_name is not None:
        st.title(f"{model_name} Playground")
    else:
        st.title("Playground")
    
    text_placeholder = st.empty()

    text_input = st.sidebar.text_area("Enter your query:")

    col11, col12 = st.sidebar.columns([0.4, 1])

    with col11:
        button = st.empty()
    with col12:
        reset = st.button("Clear", on_click=lambda: client.clear_messages())

    with st.sidebar.expander("Generation Settings:", expanded=False):
        do_sample = st.checkbox("do_sample", True)

        col21, col22, col23 = st.columns(3)

        with col21:
            temperature = st.slider("temperature", 0.0, 1.0, 1.0, 0.01, disabled=not do_sample)
        with col22:
            top_p = st.slider("top-p", 0.0, 1.0, 1.0, 0.01, disabled=not do_sample)
        with col23:
            max_length = st.slider("max_length", 1, max_output_length, max_output_length, 1)

    if 'request_id' not in st.session_state:
        st.session_state['request_id'] = ''
    if 'response_text' not in st.session_state:
        st.session_state['response_text'] = ''
    
    def stop_callback():
        if 'request_id' not in st.session_state:
            st.session_state['response_text'] = ''
        response = client.stop_stream(st.session_state['request_id'])
        if response['status'] == 'error':
            print('\n__ERROR__:', response['error_message'])
            st.error(response['error_message'])
        elif response['status'] == 'success':
            print("\nSTOPPED:", response['data'])
        else:
            raise NotImplementedError
    
    def submit_callback():
        button.button("Stop", on_click=stop_callback)

        client.add_user_message(text_input)
        sse_client = client.get_model_response_stream(
            seed=None, 
            max_input_length=max_input_length, 
            max_new_tokens=max_length, 
            do_sample=do_sample, 
            num_beams=1, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=None, 
            eos_token_id=eos_token_id, 
        )
        if isinstance(sse_client, str):
            print('__ERROR__:', sse_client)
            st.error(sse_client)
            return
        
        # display token stream
        def token_stream_read(request_id: str, full_texts: List[str], next_tokens: List[str]):
            st.session_state['request_id'] = request_id
            st.session_state['response_text'] = full_texts[0]
            print(client.postproc_model_response(next_tokens[0], lstrip_space=False), end='', flush=True)
            raw_chat_text = format_chat_text(client.postproc_model_response(st.session_state['response_text']))
            text_placeholder.markdown(raw_chat_text.replace('\n', '  \n'), unsafe_allow_html=True)
        def token_stream_error(e: str):
            print('\n__ERROR__:', e)
            st.error(e)
        print('User:', text_input)
        print('Koala:', end='', flush=True)
        map_token_stream(
            sse_client, 
            token_stream_read, 
            error_callback=token_stream_error, 
        )
        print()
        client.add_model_response(client.postproc_model_response(st.session_state['response_text']))

    def submit_callback_no_stream():
        client.add_user_message(text_input)
        response = client.get_model_response(
            seed=None, 
            max_input_length=max_input_length, 
            max_new_tokens=max_length, 
            do_sample=do_sample, 
            num_beams=1, 
            temperature=temperature, 
            top_p=top_p, 
            top_k=None, 
            eos_token_id=eos_token_id, 
        )

        if response['status'] == 'error':
            print('\n__ERROR__:', response['error_message'])
            st.error(response['error_message'])
        elif response['status'] == 'success':
            st.session_state['response_text'] = response['data'][0]
            print('User:', text_input)
            print('Koala:', response['data'][0])
            client.add_model_response(client.postproc_model_response(st.session_state['response_text']))
            text_placeholder.markdown(format_chat_text(None).replace('\n', '  \n'), unsafe_allow_html=True)
        else:
            raise NotImplementedError
    
    if stream:
        button.button("Submit", on_click=submit_callback)
    else:
        button.button("Submit", on_click=submit_callback_no_stream)

    if len(client.user_messages) == (len(client.model_responses)+1) and st.session_state['response_text'] != '':
        client.add_model_response(client.postproc_model_response(st.session_state['response_text']))

    text_placeholder.markdown(format_chat_text(None).replace('\n', '  \n'), unsafe_allow_html=True)

if __name__ == "__main__":
    # example usage:
    # streamlit run streamlit_koala_client.py -- --host http://my-host:8000
    tyro.cli(main)
