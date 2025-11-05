# chatbot-project
**INSTRUCTIONS TO SET UP AND RUN THE CHATBOT PROJECT**
1. **Download Ollama** from https://ollama.com/download/windows and install it.

2. **Clone the repo** 
    ```bash
    git clone https://github.com/TranGiaHung30302/chatbot-project.git
    ```

3. **Navigate to the project directory**:
   ```bash
   cd chatbot-project
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

5. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

6. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
7. **Pull the llama model**
    ```bash
    ollama pull llama3.2:3b
    ```

8. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```