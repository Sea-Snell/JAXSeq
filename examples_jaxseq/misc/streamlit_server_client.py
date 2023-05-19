import tyro
import streamlit as st
from typing import Optional, Tuple, List, Union
from examples_jaxseq.misc.commandline_server_client import Client
from JaxSeq.utils import strip_prompt_from_completion
import json
from JaxSeq.stream_tokens import map_token_stream

def main(
    host: Union[str, List[str]]='http://127.0.0.1:8000/', 
    max_input_length: int=512, 
    max_output_length: int=512, 
    model_name: Optional[str]=None, 
    eos_token_id: Optional[int]=None, 
    remove_prompt_from_generation: bool=False, 
    stream: bool=True, 
):
    if isinstance(host, str):
        host = [host]
    client = Client(hosts=host)

    if model_name is not None:
        st.title(f"{model_name} Playground")
    else:
        st.title("Playground")

    with st.expander("Generation Settings:", expanded=True):
        do_sample = st.checkbox("do_sample", True)

        col11, col12, col13 = st.columns(3)

        with col11:
            temperature = st.slider("temperature", 0.0, 1.0, 1.0, 0.01, disabled=not do_sample)
        with col12:
            top_p = st.slider("top-p", 0.0, 1.0, 1.0, 0.01, disabled=not do_sample)
        with col13:
            max_length = st.slider("max_length", 1, max_output_length, max_output_length, 1)

    text_input = st.text_area("Enter your query:")

    button = st.empty()
    
    text_placeholder = st.empty()

    if 'request_id' not in st.session_state:
        st.session_state['request_id'] = ''
    if 'response_text' not in st.session_state:
        st.session_state['response_text'] = ''
    if 'submitted_text_input' not in st.session_state:
        st.session_state['submitted_text_input'] = ''
    
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
        st.session_state['submitted_text_input'] = text_input

        sse_client = client.generate_stream(
            [text_input], 
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
            print(next_tokens[0], end='', flush=True)
            output_text = st.session_state['response_text']
            if remove_prompt_from_generation:
                output_text = strip_prompt_from_completion(st.session_state['submitted_text_input'], output_text)
            text_placeholder.markdown(output_text.replace('\n', '  \n'), unsafe_allow_html=True)
        def token_stream_error(e: str):
            print('\n__ERROR__:', e)
            st.error(e)
        map_token_stream(
            sse_client, 
            token_stream_read, 
            error_callback=token_stream_error, 
        )
        print()
    
    def submit_callback_no_stream():
        st.session_state['submitted_text_input'] = text_input

        response = client.generate(
            [text_input], 
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
            print(response['data'][0])
            output_text = st.session_state['response_text']
            if remove_prompt_from_generation:
                output_text = strip_prompt_from_completion(st.session_state['submitted_text_input'], output_text)
            text_placeholder.markdown(output_text.replace('\n', '  \n'), unsafe_allow_html=True)
        else:
            raise NotImplementedError

    if stream:
        button.button("Submit", on_click=submit_callback)
    else:
        button.button("Submit", on_click=submit_callback_no_stream)

    output_text = st.session_state['response_text']
    if remove_prompt_from_generation:
        output_text = strip_prompt_from_completion(st.session_state['submitted_text_input'], output_text)
    text_placeholder.markdown(output_text.replace('\n', '  \n'), unsafe_allow_html=True)

if __name__ == "__main__":
    # example usage:
    # streamlit run streamlit_server_client.py -- --host http://my-host:8000 --max-input-length 512 --max-output-length 512 --model-name T5
    tyro.cli(main)
