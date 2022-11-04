build-client:
	cd frontend && npm install && PUBLIC_DEV_MODE=PROD npm run build && rm -rf ../static && cp -r build/ ../static/
build-dev:
	cd frontend && npm install && PUBLIC_DEV_MODE=DEV npm run build-dev && rm -rf ../static && cp -r build/ ../static/
run-front-dev:
	cd frontend && npm install && PUBLIC_DEV_MODE=DEV npm run dev
run-dev:
	rm -rf .data/ && FLASK_DEBUG=development python app.py
run-prod:
	python app.py
build-all: run-prod