up:
	docker compose -d --build

down:
	docker compose down

run-local:
	APP_ENV=local poetry run uvicorn app.main:app --reload
prune:
	docker system prune -a --volumes -f
