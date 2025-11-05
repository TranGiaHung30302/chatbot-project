# chatbot-project

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Pull the llama model**
    ```bash
    ollama pull llama3.2:3b
    ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```