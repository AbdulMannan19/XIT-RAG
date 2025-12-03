eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Dipy3XwvZ9HewzNDMrpKPnKfGXfoUu3NP4ZUZND3hKw

python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

git status
git add .
git config user.email "abdulmannan34695@gmail.com"
git config user.name "AbdulMannan19"
git commit -m "quick commit"
git push


Controllers - Handling HTTP Requests
Services - Core functions of the business logic 
Handlers - Orchestrating those functions to perform the API call
Models - Request, response and other schemas
Helpers - Helper functions for the Services folder (domain specific delegated logic)
Utils - Cross Cutting simple functions used accross different domains

All constants in respective files on the top of the code
env file only has private/secret variables like qdrant url adn API key (Ollama url of VM must also be added)

