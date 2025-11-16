.PHONY: setup run eval

setup:
\tpip install -r requirement.txt
\techo "Downloading DIC PDFs..."
\tmkdir -p data
\tcurl -L -o data/dataset_eval.zip https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/447dd4/dataset_eval.zip
\tunzip -d data/eval data/dataset_eval.zip
\trm data/dataset_eval.zip
\tcurl -L -o data/DIC.zip https://blent-learning-user-ressources.s3.eu-west-3.amazonaws.com/projects/447dd4/DIC.zip
\tunzip -d data ../data/DIC.zip
\trm data/DIC.zip

run:
\tuvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 3

eval:
\tpython src/evaluation.py
