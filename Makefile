init:
	code .
	poetry config --local virtualenvs.create true
	poetry config --local virtualenvs.in-project true
	poetry env use 3.11
	poetry install
	poetry update
	git init
	echo ".venv/" >> .gitignore
	echo "logs/" >> .gitignore
	git lfs install
	git lfs track "*.fit"
	git lfs track "*.hdf5"
	git lfs track "*.html"
	git lfs track "*.parquet"
	poetry export -f requirements.txt --output requirements.txt
	git add .
	poetry run pre-commit run --all-files
	git commit -am "Initial commit after initializing the project."
	poetry shell

test:
	pytest .

coverage:
	pytest --cov=blog tests/


publish:
	poetry run quarto render
	git add altair/
	git add docs/
	git commit -am "Updating blog."
	git push
