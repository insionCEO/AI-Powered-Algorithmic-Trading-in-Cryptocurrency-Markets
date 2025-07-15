docker login azure --client-id 48d135de-93c3-4a37-9f98-eed3375afd67 --client-secret ZOi8Q~u1zWvEtgyKPELxuIEvvLvA_yxnk.TTebcM --tenant-id bdeaeda8-c81d-45ce-863e-5232a535b7cb

az acr login --name tradeaicr 

docker context use default 

docker-compose --project-name tradeai build

docker-compose --project-name tradeai push

docker context use tradeai_context 

docker-compose --project-name tradeai up 