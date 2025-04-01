import streamlit as st
import base64
import random
import time
from langchain.llms import CTransformers
from langchain import PromptTemplate
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import speech_recognition as sr
from pathlib import Path
import logging
from functools import lru_cache
import sounddevice as sd
import wavio
import numpy as np
from tempfile import NamedTemporaryFile
import os

class BlogGeneratorApp:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state for voice input
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
        
        # Validate model file existence
        self.model_path = Path('llama-2-7b-chat.ggmlv3.q8_0.bin')
        if not self.model_path.exists():
            self.logger.error(f"Model file not found at {self.model_path}")
            
        # Set page configuration
        st.set_page_config(
            page_title="GEN AI content generator", 
            page_icon="🚀", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize LLM once during startup
        try:
            self.llm = CTransformers(
                model=str(self.model_path),
                model_type='llama',
                config={
                    'max_new_tokens': 256,
                    'temperature': 0.01
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
        
        # Custom CSS for enhanced styling
        self.local_css()
    
    def local_css(self):
        st.markdown("""
        <style>
        /* Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Glassmorphic Card Design */
        .main-container {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.125);
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Elegant Input Styles */
        .stTextInput > div > div > input, 
        .stSelectbox > div > div > select {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            padding: 10px;
            transition: all 0.3s ease;
        }
        
        /* Recording Animation */
        @keyframes recording-pulse {
            0% { background-color: rgba(255, 0, 0, 0.2); }
            50% { background-color: rgba(255, 0, 0, 0.5); }
            100% { background-color: rgba(255, 0, 0, 0.2); }
        }
        
        .recording-active {
            animation: recording-pulse 1s infinite;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Rest of your existing CSS styles */
        .stTextInput > div > div > input:focus, 
        .stSelectbox > div > div > select:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            font-weight: bold;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .generated-text {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
        }
        
        .css-1aumxhk {
            background: rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }
        </style>
        """, unsafe_allow_html=True)

    @lru_cache(maxsize=100)
    def generate_llama_blog(self, input_text, no_words, blog_style):
        try:
            if not self.llm:
                st.error("LLM model not properly initialized")
                return None
            
            if not input_text or not blog_style:
                raise ValueError("Input text and blog style are required")
            
            if not isinstance(no_words, int) or no_words < 100 or no_words > 1000:
                raise ValueError("Word count must be between 100 and 1000")
                
            template = """
            Write a {blog_style} style blog post about {input_text}
            within {no_words} words. Focus on providing valuable insights 
            and maintaining a consistent tone throughout the content.
            """
            
            prompt = PromptTemplate(
                input_variables=["blog_style", "input_text", 'no_words'],
                template=template
            )
            
            response = self.llm(prompt.format(
                blog_style=blog_style, 
                input_text=input_text, 
                no_words=no_words
            ))
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating blog: {e}")
            st.error(f"Error generating blog: {str(e)}")
            return None

    def record_voice(self):
        """Record voice input and convert to text using SpeechRecognition"""
        try:
            # Audio recording parameters
            duration = 5  # seconds
            sample_rate = 44100
            channels = 1
            
            # Create a progress bar for the countdown
            progress_bar = st.progress(0)
            countdown_text = st.empty()
            
            # Countdown
            for i in range(3, 0, -1):
                countdown_text.info(f"🎙️ Recording will start in {i} seconds...")
                progress_bar.progress((3 - i) / 3)
                time.sleep(1)
            
            # Show recording status
            status_text = st.empty()
            status_text.markdown(
                '<div class="recording-active">🔴 Recording... Speak now!</div>',
                unsafe_allow_html=True
            )
            
            # Record audio
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float64'
            )
            sd.wait()
            
            status_text.info("Processing your speech...")
            
            # Create a temporary WAV file
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                wavio.write(temp_audio.name, recording, sample_rate, sampwidth=2)
                
                # Initialize speech recognizer
                recognizer = sr.Recognizer()
                
                # Read the temporary file and convert to text
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        status_text.success(f"📝 Transcribed Text: {text}")
                        return text
                    except sr.UnknownValueError:
                        status_text.warning("🤔 Could not understand the audio. Please try again.")
                    except sr.RequestError as e:
                        status_text.error(f"🚫 Error with the speech recognition service; {e}")
                    finally:
                        # Clean up
                        progress_bar.empty()
                        countdown_text.empty()
                        os.unlink(temp_audio.name)
                        
        except Exception as e:
            self.logger.error(f"Error in voice recording: {e}")
            st.error("Error recording audio. Please check your microphone settings.")
        
        return ""

    def create_blog_metrics_visualization(self):
        try:
            metrics_data = {
                'Blog Style': ['Technical', 'Professional', 'Casual', 'Academic', 'Creative'],
                'Average Words': [350, 400, 300, 450, 250],
                'Popularity': [85, 90, 75, 65, 95]
            }
            
            df = pd.DataFrame(metrics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    df, 
                    x='Blog Style', 
                    y='Average Words', 
                    title='Average Blog Length by Style',
                    color='Blog Style',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig1)
            
            with col2:
                fig2 = go.Figure(data=go.Scatterpolar(
                    r=df['Popularity'],
                    theta=df['Blog Style'],
                    fill='toself'
                ))
                fig2.update_layout(
                    title='Blog Style Popularity',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig2)
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            st.error("Error creating visualizations")

    def get_download_link(self, text, filename):
        try:
            b64 = base64.b64encode(text.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="display:block;text-align:center;background:rgba(255,255,255,0.1);color:white;padding:10px;border-radius:10px;text-decoration:none;">📥 Download Blog</a>'
        except Exception as e:
            self.logger.error(f"Error creating download link: {e}")
            return None

    def sanitize_input(self, text):
        return text.strip()

    def run(self):
        try:
            st.markdown("""
            <h1 style='text-align: center; color: white; 
            text-shadow: 0 0 10px rgba(255,255,255,0.5), 
            0 0 20px rgba(255,255,255,0.3);
            animation: pulse 2s infinite;'>
            🚀 Offline content generator
            </h1>
            <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                input_text = st.text_input(
                    "Content Topic",
                    value=st.session_state.get('input_text', ''),
                    help="Enter your blog topic."
                )
                input_text = self.sanitize_input(input_text)
                
                # Voice recording section
                st.markdown("### Voice Input")
                st.markdown("Click below to record your topic instead of typing")
                if st.button("🎤 Record Voice"):
                    recorded_text = self.record_voice()
                    if recorded_text:
                        st.session_state.input_text = recorded_text
                        st.experimental_rerun()
            
            with col2:
                blog_styles = {
                    '🖥️ Technical': 'technical', 
                    '💼 Professional': 'professional', 
                    '😎 Casual': 'casual', 
                    '🎓 Academic': 'academic', 
                    '🎨 Creative': 'creative'
                }
                blog_style = st.selectbox("Blog Style", list(blog_styles.keys()))
                blog_style = blog_styles[blog_style]
            
            no_words = st.slider(
                "📏 Word Count", 
                min_value=100, 
                max_value=1000, 
                value=250, 
                step=50
            )
            
            if st.button("✨ Generate Magical content ✨"):
                if not input_text:
                    st.warning("🚨 Please enter a topic!")
                else:
                    with st.spinner('Generating content...'):
                        generated_blog = self.generate_llama_blog(
                            input_text, 
                            no_words, 
                            blog_style
                        )
                        if generated_blog:
                            st.markdown('<div class="generated-text">', 
                                      unsafe_allow_html=True)
                            st.write(generated_blog)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            download_link = self.get_download_link(
                                generated_blog, 
                                f"{input_text.replace(' ', '_')}_blog.txt"
                            )
                            if download_link:
                                st.markdown(download_link, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.header("📊 Content Generation Insights")
            self.create_blog_metrics_visualization()

        except Exception as e:
            self.logger.error(f"Error in main application loop: {e}")
            st.error("An unexpected error occurred. Please try again.")

def main():
    try:
        app = BlogGeneratorApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        logging.error(f"Application initialization failed: {e}")

if __name__ == "__main__":
    main()
