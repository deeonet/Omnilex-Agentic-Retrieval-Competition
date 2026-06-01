## Project setup

1) Create a Kaggle account if you do not already have one

2) Go to the Rules tab of the  competition and acept all terms by joining the competition

3) Generate and securely store an API token for your kaggle account

4) Clone the forked repo: git clone git@github.com:deeonet/Omnilex-Agentic-Retrieval-Competition.git && cd Omnilex-Agentic-Retrieval-Competition

5) Create a dev branch for yourself: git checkout -b branch_name

6) Setup & activate virtual environment: python3 -m venv .venv && source .venv/bin/activate

7) Install requirements: python3 -m pip install -r requirements-dev.txt

8) Set kaggle env variable using value from step 3: export KAGGLE_API_TOKEN=KGAT_generated_token

9) Download data: python3 -m download_kaggle_data

10) This will print the path where the data is saved: mv printed_path data

11) python utils/download_data.py
