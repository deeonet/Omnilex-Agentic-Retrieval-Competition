import kagglehub

kagglehub.login()
# Download latest version
path = kagglehub.competition_download('llm-agentic-legal-information-retrieval')

print("Path to competition files:", path)
