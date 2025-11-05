# chatbot-project
**INSTRUCTIONS TO SET UP AND RUN THE CHATBOT PROJECT**
1. **Download Ollama** from https://ollama.com/download/windows and install it.

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Pull the llama model**
    ```bash
    ollama pull llama3.2:3b
    ```

6. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```