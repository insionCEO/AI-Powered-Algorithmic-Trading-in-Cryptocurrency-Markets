dvc pull data/processed/10-min-samples/sample-1-BTC.csv

dvc pull data/processed/10-min-samples/sample-2-BTC.csv

docker context use default 

docker-compose --project-name tradeai build

docker-compose --project-name tradeai up

docker cp data/processed/10-min-samples/sample-1-BTC.csv tradeai-server-1:app/data/processed/10-min-samples/

docker cp data/processed/10-min-samples/sample-2-BTC.csv tradeai-server-1:app/data/processed/10-min-samples/