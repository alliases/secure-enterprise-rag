up:
	docker compose --env-file .env.docker up -d --build

down:
	docker compose --env-file .env.docker down

run-local:
	APP_ENV=local poetry run uvicorn app.main:app --reload
