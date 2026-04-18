FOR /F "tokens=5" %%i IN ('netstat -aon ^| findstr ":8002" ^| findstr "LISTENING"') DO (
    echo Killing process with PID %%i on port 8002.
    taskkill /F /PID %%i
)
pip install -r requirements.txt
python proxy_server.py
